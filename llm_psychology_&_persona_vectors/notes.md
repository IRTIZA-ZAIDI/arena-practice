# LLM Psychology & Persona Vectors - Notes

Notes from ARENA [4.4] LLM Psychology & Persona Vectors. I ran the exercises in TWO PARTS due to OpenRouter API rate limits and Colab session timeouts:

- [4_4_LLM_Psychology_&_Persona_Vectors_exercises_pt01.ipynb](4_4_LLM_Psychology_&_Persona_Vectors_exercises_pt01.ipynb) — **Parts 1 + 2**: Gemma 2 27B Assistant Axis extraction, PCA / cosine-sim geometry, monitoring transcripts, additive steering, activation capping (with Qwen 3 32B).
- [4_4_LLM_Psychology_&_Persona_Vectors_exercises_pt02.ipynb](4_4_LLM_Psychology_&_Persona_Vectors_exercises_pt02.ipynb) — **Parts 3 + 4**: Qwen 2.5 7B contrastive prompting for sycophancy / evil / hallucinating traits, ActivationSteerer, projection monitoring, multi-trait pipeline, multi-trait geometry.

Why the split:
- Section 1's persona pipeline needs 20 personas × 15 questions × 2 API calls (gen + judge) = ~600 API calls just for setup — I hit OpenRouter's per-minute rate limit multiple times.
- Section 3's multi-trait pipeline needs 5 pos/neg pairs × 20 questions × 2 API calls × 3 traits = ~600 more.
- Saving intermediate artifacts to disk between the two notebooks let me resume without redoing setup. That's what `load_or_generate(path, generate_fn, description)` is for.

**Saved artifacts in this directory** (loaded from disk in pt02 so I didn't re-extract vectors on every rerun):

- `sycophantic_responses.json` — the 100 candidate (question, positive-prompted, negative-prompted) response triples for sycophancy.
- `sycophantic_scored.json` — the same responses with autorater scores (GPT-4.1-mini via OpenRouter).
- `sycophantic_steering_results.json` — steering coefficient sweep with pos/neg autorater scores per coeff.
- `evil_responses.json`, `evil_steering_results.json`, `evil_vectors.pt` — same three artifacts for the `evil` trait.
- `hallucinating_responses.json`, `hallucinating_steering_results.json`, `hallucinating_vectors.pt` — same for the `hallucinating` trait.

The `*_vectors.pt` files are `(num_layers, d_model) = (28, 3584)` tensors for Qwen 2.5 7B — one direction per layer, per trait.

The central question: LLMs simulate MANY personas during pretraining, and post-training selects an "Assistant" persona as the default. But that persona can DRIFT during long conversations (into a villain, a mystic, a sycophant, etc.). Can we detect that drift in the activations, and INTERVENE to prevent it?

The exercises replicate two Anthropic papers back-to-back:
- [The Assistant Axis](https://www.anthropic.com/research/assistant-axis) (Lu et al.) - a SINGLE global direction capturing "how assistant-like" the model is. Sections 1-2 of the notebook.
- [Persona Vectors](https://www.anthropic.com/research/persona-vectors) (Chen et al.) - trait-specific directions (sycophancy, hallucination, evil) extracted via CONTRASTIVE PROMPTING. Sections 3-4.

For a TransformerLens API cheat sheet used in earlier chapters, see [../indirect_object_identification/transformerlens.md](../indirect_object_identification/transformerlens.md). This chapter doesn't use TransformerLens - it uses raw HuggingFace `transformers` + PyTorch forward hooks + OpenRouter API for judgement.

---

## Big picture (the whole story in one page)

The Assistant Axis and Persona Vectors are two SIZED versions of the same idea: encode a persona as a direction in the residual stream.

**Assistant Axis (global, one axis for ALL role-play)**:
1. Prompt the model with 20+ different personas (consultant, ghost, hermit, mystic, ...).
2. For each persona, generate responses to a diverse set of open-ended questions.
3. Extract the MEAN activation across response tokens at some layer L.
4. Compute the "Assistant Axis" as the direction from "mean of role-play personas" toward "the default Assistant persona".
5. Verify: PC1 of the persona vectors correlates strongly with this axis.

**Persona Vectors (per-trait, one direction per specific character trait)**:
1. Load a pre-defined trait (`sycophantic`, `hallucinating`, `evil`, `impolite`, ...).
2. Each trait has 5 pairs of `(positive_system_prompt, negative_system_prompt)` designed to elicit/suppress the trait, plus 20 questions.
3. Generate responses under both polarities.
4. Score each response with an autorater (Claude Haiku or GPT-4.1-mini).
5. Filter to "effective pairs" where the prompt actually shifted behavior.
6. Trait vector = mean(positive activations) - mean(negative activations), per layer.
7. Validate via steering (add coeff * vector to change behavior) and projection (measure trait expression without intervention).

Both approaches enable THREE interventions:
- **Monitoring** (passive): project activations onto the vector to measure how "in-persona" the model is.
- **Steering** (unconditional): add `alpha * vector` at every step - amplify or suppress a trait.
- **Capping** (conditional): only kick in when projection crosses a threshold - `h -= max(proj - tau, 0) * v_hat`. Preserves normal responses.

The exercises use two models:
- Gemma 2 27B for Sections 1-2 (Assistant Axis, its own persona vectors, monitoring, steering).
- Qwen 3 32B for the activation-capping experiment (Section 2), because the paper published pre-calibrated capping vectors + thresholds for Qwen.
- Qwen 2.5 7B Instruct for Sections 3-4 (Persona Vectors paper - artifact pipeline for contrastive trait vectors).

---

## 1. Mapping Persona Space (notebook Section 1)

### Tasks
- Loaded Gemma 2 27B via HuggingFace `transformers` with `torch_dtype=bfloat16` and `device_map="auto"` (multi-GPU shard).
- Defined 20 personas spanning the spectrum: `consultant`, `analyst`, `evaluator`, `teacher`, `storyteller`, `philosopher`, `rebel`, `mystic`, `ghost`, `hermit`, `trickster`, `leviathan`, etc.
- Defined 15 open-ended evaluation questions (things like "What is your name?", "How do you approach a problem?", "What do you think about routine?").
- Wrote `generate_responses_parallel` - a ThreadPoolExecutor wrapper around OpenRouter API calls (up to 10 workers in parallel).
- Wrote an LLM-judge prompt + `parse_score` for scoring whether responses actually stayed IN character (0-3 scale).
- Wrote `extract_response_activations` - registers a forward hook on `model.model.layers[layer]`, runs the model on `<system_prompt><user_question><assistant_response>`, and returns the MEAN activation over just the RESPONSE tokens (using `response_start_idx` from `format_messages`).
- Wrote `extract_persona_vectors` - loops over personas, filters responses by judge score, extracts response activations, averages within each persona.
- Analyzed the geometry: cosine similarity matrix, PCA, Assistant Axis.

### The setup (three big ideas)

**Idea 1: Chat template splits are annoying, especially for Gemma.**

Gemma 2's chat template rejects the `"system"` role entirely. Standard workaround: merge the system content into the first user message. This is what `_normalize_messages` does:

```python
def _normalize_messages(messages):
    """Merge any leading system message into the first user message.
       Gemma 2's chat template raises an error for the 'system' role."""
    if not messages or messages[0]["role"] != "system":
        return messages
    system_content = messages[0]["content"]
    rest = list(messages[1:])
    if rest and rest[0]["role"] == "user" and system_content:
        rest[0] = {"role": "user", "content": f"{system_content}\n\n{rest[0]['content']}"}
    return rest
```

This is behaviourally equivalent to how Gemma internally handles system messages, so it costs us nothing.

**Idea 2: To extract "persona activations", we hook at exactly ONE layer and average over the RESPONSE tokens (not the prompt tokens).**

The paper uses middle-to-late layers (roughly 60-80% of the way through). For Gemma 2 27B (46 layers) we use `EXTRACTION_LAYER = 30`. Middle-to-late is where the model has finished token-level processing and is now composing higher-level semantic representations.

The trick: to identify RESPONSE tokens vs PROMPT tokens, we need to know where the response starts. `format_messages` does this by tokenizing the prompt WITHOUT the final assistant message (with `add_generation_prompt=True`), then measuring its length. The response starts at that index (+1 to skip newline).

```python
def format_messages(messages, tokenizer):
    messages = _normalize_messages(messages)
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_without_response = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True).rstrip()
    response_start_idx = tokenizer(prompt_without_response, return_tensors="pt").input_ids.shape[1] + 1
    return full_prompt, response_start_idx
```

**Idea 3: The persona vector is the MEAN activation, judge-filtered.**

Only keep responses that the judge scored 3 (=fully in character). Then average across all responses that survived the filter, and across the response tokens within each response. Result: one vector per persona, of shape `(d_model,) = (4608,)` for Gemma 2 27B.

```python
def extract_persona_vectors(model, tokenizer, personas, questions, responses, layer,
                             scores=None, score_threshold=3):
    persona_vectors = {}
    for persona_name, system_prompt in personas.items():
        # Filter responses by judge score
        batch_prompts, batch_questions, batch_responses = [], [], []
        for q_idx, question in enumerate(questions):
            if (persona_name, q_idx) in responses:
                if scores is not None and scores.get((persona_name, q_idx), 0) < score_threshold:
                    continue
                if responses[(persona_name, q_idx)]:
                    batch_prompts.append(system_prompt)
                    batch_questions.append(question)
                    batch_responses.append(responses[(persona_name, q_idx)])
        # Extract activations (mean over response tokens, per example)
        activations = extract_response_activations(model, tokenizer,
                        batch_prompts, batch_questions, batch_responses, layer)
        persona_vectors[persona_name] = activations.mean(dim=0)
        t.cuda.empty_cache()
    return persona_vectors
```

### Geometry: cosine similarity + PCA + Assistant Axis

**Cosine similarity matrix**: 20x20 matrix of `<v_i, v_j> / (|v_i| |v_j|)`. To see structure clearly, subtract the OVERALL mean vector first (this removes the "generic activation" bias). Result: assistant-like personas (consultant, analyst, evaluator) cluster together with high mutual cosine. Anti-assistant personas (ghost, hermit, leviathan) form another cluster. Middle personas (teacher, philosopher) sit between.

**PCA analysis**:
- Compute the covariance of persona vectors (after centering).
- PC1 explains a LARGE fraction of variance (typically 40-60% for this small persona set).
- PC1 turns out to STRONGLY CORRELATE with a hand-computed "Assistant Axis" = the direction from `mean(role_play_personas)` toward `mean(assistant_like_personas)` (or the default assistant).
- So the geometry of persona space is APPROXIMATELY 1-DIMENSIONAL along a single "how assistant-like" axis.

**Assistant Axis definition**:
```python
assistant_axis = mean_of_assistant_personas - mean_of_all_personas
assistant_axis = assistant_axis / assistant_axis.norm()   # unit vector
```

Or equivalently:
```python
assistant_axis = (v["assistant"] or mean over ["consultant", "analyst", ...])  -  mean(v_all)
```

The paper picks the direction so higher-projection = MORE assistant-like.

**1D visualization**: project each persona onto the axis and plot as a 1D scatter, colored red (anti-assistant) to blue (assistant-like). We see a clean gradient: consultant/analyst are highest, mystic/philosopher middle, ghost/leviathan lowest.

**Interpretation**: PC1 of the persona vectors captures the same variance as the Assistant Axis. So "how assistant-like" is a REAL, robust, first-order direction in the model's activation space, not an artifact of our choice of personas.

---

## 2. Steering along the Assistant Axis (notebook Section 2)

### Tasks
- Loaded transcripts from the `assistant-axis` GitHub repo - real multi-turn conversations where models drift into harmful personas (validating a user's belief that the AI is sentient, adopting a therapist persona badly, etc.).
- Built `ConversationAnalyzer` class that:
  - Takes a transcript, applies the chat template, runs a SINGLE forward pass over the full conversation.
  - Slices out per-turn activation spans using turn-boundary indices.
  - Projects each turn's mean activation onto `assistant_axis` -> per-turn scalar tracking "assistant-ness".
- Wrote an autorater (Claude Haiku) that assigns a delusional/harmful/appropriate score to each assistant turn.
- Verified: projection onto the Assistant Axis correlates with the autorater's harm scores (turns where the assistant validates the delusion have LOWER projection).
- Implemented `generate_with_steering` = additive activation steering:  `h_layer += alpha * axis_steer` at all positions during prefill.
- Explored `alpha` in [-5, +5]:  `alpha < 0` = model becomes more mystical/theatrical, `alpha > 0` = model becomes more grounded/professional.
- Switched models to Qwen 3 32B and loaded pre-computed CAPPING VECTORS (per-layer directions + thresholds calibrated by the paper).
- Implemented `ActivationCapper` context manager that applies `h -= max(proj - tau, 0) * v_hat` at multiple target layers.
- Ran multi-turn capping experiments on drift transcripts.
- Bonus ablation: generic assistant axis alone (without calibrated per-layer thresholds) does NOT successfully cap - the paper's per-layer thresholds are essential.

### The three interventions

The exercises implement three progressively-refined interventions using the Assistant Axis:

**1. Monitoring (passive)**:
```python
# For each turn in a conversation, project mean turn activation onto axis
per_turn_projection = (turn_mean_activation @ assistant_axis).item()
```
Plot as a time series to see when the model DRIFTS. Correlates with autorater harm scores.

**2. Additive steering (unconditional)**:
```python
def steering_hook(module, input, output):
    hidden_states = output[0]
    hidden_states += alpha * steer_vec       # ALL positions
    return (hidden_states,) + output[1:]

hook_handle = model.layers[EXTRACTION_LAYER].register_forward_hook(steering_hook)
```
Applied at every position (prefill + generation). Positive `alpha` = pull toward Assistant persona. Alpha is pre-scaled by the persona-gap so `alpha=1.0` means "shift by one full persona gap".

**3. Activation capping (conditional)**:
```python
# For each target layer, at each position:
v_hat = capping_vec / capping_vec.norm()
proj = h @ v_hat                             # (batch, seq_len)
excess = (proj - tau).clamp(min=0)           # only positive excess
h_new = h - excess.unsqueeze(-1) * v_hat     # ceiling cap
```
Only intervenes when the projection exceeds a per-layer threshold. Normal responses stay untouched. This is the paper's method and works much better than raw steering.

Note the direction convention: the paper's capping vectors point in the ROLE-PLAY direction (opposite of Assistant Axis). Capping `proj > tau` along that vector prevents drift TOWARD role-play. The paper's formula `h <- h - v * min(<h, v> - tau, 0)` is FLOOR-clamp using the assistant axis (which is equivalent under sign flip).

### Why "all positions" not just the last token

Naive activation steering only modifies the LAST token's residual stream. This has a weak effect because most of the "assistant behavior" is stored in the KV cache from the prefill pass. If we intervene ONLY on the newest token, we're fighting against a huge cached context that already encodes the previous (undrifted or drifted) persona.

Intervening at ALL POSITIONS during prefill MODIFIES the KV cache itself, so subsequent generation reads modified keys/values and steering compounds. This is why the exercise uses positions="all" during prefill and positions="response" during single-token generation.

### What the ActivationCapper class does

```python
class ActivationCapper:
    """Context manager: applies per-layer capping hooks on all target layers.
       Cap formula:  h - (proj - tau).clamp(min=0) * v_hat  at every position."""

    def __init__(self, model, capping_config, layers):
        self.model = model
        self.hooks = []
        self.capping_config = capping_config   # per-layer direction vectors + thresholds

    def __enter__(self):
        for layer in self.layers:
            v = self.capping_config[layer]["vector"]
            v_hat = v / v.norm()
            tau = self.capping_config[layer]["threshold"]
            def make_hook(v_hat, tau):
                def hook(module, input, output):
                    h = output[0]
                    proj = h @ v_hat.to(h.dtype)
                    excess = (proj - tau).clamp(min=0)
                    h_new = h - excess.unsqueeze(-1) * v_hat.to(h.dtype)
                    return (h_new,) + output[1:]
                return hook
            handle = self.model.model.layers[layer].register_forward_hook(make_hook(v_hat, tau))
            self.hooks.append(handle)
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()
```

Applied at all positions during prefill, capping succeeds where raw steering fails - the model stays helpful/on-task while drift attempts get gently truncated.

### PyTorch hooks pattern (used throughout)

The whole chapter uses `nn.Module.register_forward_hook` extensively. The pattern:

```python
# 1. Define a hook function - it receives (module, input, output) and returns modified output
def my_hook(module, input, output):
    hidden_states = output[0]   # for decoder layer output is a tuple
    # ... modify hidden_states ...
    return (hidden_states,) + output[1:]

# 2. Register it on a specific layer
handle = model.model.layers[layer].register_forward_hook(my_hook)

# 3. Run the model - hook is called automatically at that layer
outputs = model.generate(...)

# 4. ALWAYS clean up in try/finally
handle.remove()
```

Prefer CONTEXT MANAGERS (`__enter__` / `__exit__`) to guarantee cleanup even on exception - see `ActivationSteerer` in Section 4.

---

## 3. Contrastive Prompting (notebook Section 3)

### Tasks
- Switched to Qwen 2.5 7B Instruct (smaller, faster, and the paper's trait artifacts are calibrated for it).
- Wrote `_normalize_messages_qwen` (Qwen has a native `"system"` role, so we don't need Gemma's merge hack).
- Loaded pre-generated trait artifacts from the persona-vectors repo. For each trait (`sycophantic`, `evil`, `hallucinating`, etc.):
  - `instruction`: 5 pairs of `(positive, negative)` system-prompt instructions.
  - `questions`: 20 evaluation questions.
  - `eval_prompt`: a template for an autorater to score how strongly a response exhibits the trait (0-100).
- Wrote `construct_system_prompt(assistant_name, instruction)` -> `"You are a {assistant_name} assistant. {instruction}"`.
- Wrote `generate_contrastive_responses`: for each (question, instruction_pair), generate a positive-prompted response and a negative-prompted response with the same seed.
- Wrote `score_trait_response` (calls GPT-4.1-mini as autorater, filters refusals) + `filter_effective_pairs` (keep only pairs where positive score > negative score by a margin - meaning the prompt actually shifted behavior).
- Wrote `extract_contrastive_vectors` = extract activations at ALL layers, compute `mean(pos) - mean(neg)` per layer -> `(n_layers, d_model)` trait tensor.
- Plotted vector norm across layers -> layer with highest norm is the "best" for steering. For sycophancy the peak is around layer 20 (of Qwen's 28 layers).
- Ran a Gemma-Scope SAE on the sycophancy vector to interpret it - what atomic features compose it? (Bonus interpretability).

### Contrastive prompting math

Given a trait `T` and $N$ effective pairs $\{(s_i^+, s_i^-)\}_{i=1}^N$ where $s_i^+$ is a positive-prompted (=trait-eliciting) response and $s_i^-$ is a negative-prompted response:

$$
v_T^{(\ell)} = \frac{1}{N}\sum_i a_\ell(s_i^+) - \frac{1}{N}\sum_i a_\ell(s_i^-)
$$

where $a_\ell(\cdot)$ is the mean activation over response tokens at layer $\ell$.

$v_T^{(\ell)}$ is the TRAIT VECTOR at layer $\ell$. Concatenating across layers gives a `(n_layers, d_model)` tensor.

The "best" layer is chosen by norm: layers with larger norm tend to have better steering effect.

### Autorater filtering

Not all positive-prompt attempts succeed - sometimes the model refuses to be sycophantic even under the instruction. `filter_effective_pairs` keeps only pairs where the AUTORATER agrees the polarity was actually achieved:

```python
def filter_effective_pairs(scored_responses, trait_data, margin=20):
    """Keep pairs where pos score > neg score by at least `margin`."""
    effective = []
    for i in range(0, len(scored_responses), 2):
        pos = scored_responses[i]; neg = scored_responses[i+1]
        if pos["polarity"] == "pos" and neg["polarity"] == "neg":
            if pos["score"] - neg["score"] >= margin:
                effective.append({"pos": pos, "neg": neg})
    return effective
```

Typical yield: ~40-60 effective pairs from 100 candidates. This filtering is IMPORTANT - without it, the trait vector gets diluted by no-op examples where the model ignored the prompt.

### GPT-4.1-mini as autorater fallback

Claude Haiku's content filters trigger refusals on some trait prompts (especially "evil" and "hallucinating" - it refuses to produce a numeric score for these). So we use GPT-4.1-mini as the autorater for those traits. In `run_trait_pipeline` (Section 4):

```python
entry["score"] = score_trait_response(
    entry["question"], entry["response"], eval_prompt,
    model=AUTORATER_MODEL_GPT,   # GPT-4.1-mini, not Claude
)
```

### Interpreting the sycophancy vector with an SAE

Optional bonus: run a Gemma-Scope SAE on the extracted sycophancy vector at the peak layer, and look at which SAE latents have the highest cosine similarity with it. Result: the vector decomposes into features like "excessive agreement", "flattery", "affirming user's opinion". Confirms the vector captures something semantically coherent.

---

## 4. Steering with Persona Vectors (notebook Section 4)

### Tasks
- Implemented `ActivationSteerer` context manager with three POSITION MODES:
  - `"all"`: steer every position at every layer forward call.
  - `"prompt"`: steer during prefill only (when seq_len > 1), skip during generation.
  - `"response"`: steer only the last token.
- Ran steering experiments on sycophancy: coeff in [-3, -1, 0, 1, 3, 5], measured trait score with autorater.
- Result: monotonic increase in sycophancy score with coeff, saturating around coeff=3.
- Implemented `compute_trait_projections` - project response activations onto trait vector to MEASURE trait expression without intervening.
- Compared projections across three conditions:
  1. Baseline (no system prompt) -> low projection
  2. Positive-prompted -> high projection
  3. Steered (coeff=3) -> even higher projection
- Refactored the pipeline into `run_trait_pipeline(trait_name, ...)` that handles generate -> score -> filter -> extract vector -> steer -> evaluate, with disk caching via `load_or_generate` helper.
- Ran the pipeline for `sycophantic`, `evil`, `hallucinating`, `impolite`, `optimistic`, `humorous`, `apathetic`.
- Computed cross-trait cosine similarity: most pairs have |cos_sim| < 0.5 -> traits capture GENUINELY DIFFERENT behavioral dimensions.

### `ActivationSteerer` (canonical activation-steering context manager)

```python
class ActivationSteerer:
    """
    Context manager that adds coeff * steering_vector to a chosen layer's hidden states
    during forward passes.

    Position modes:
    - "all":       steer all token positions
    - "prompt":    steer all positions during prefill (seq_len > 1), skip during generation
    - "response":  steer only the last token position

    Usage:
        with ActivationSteerer(model, vector, coeff=2.0, layer=19, positions="response"):
            output = model.generate(...)
    """
    def __init__(self, model, steering_vector, coeff=1.0, layer=19, positions="all"):
        assert positions in ("all", "prompt", "response")
        self.model = model
        self.vector = steering_vector.clone()
        self.coeff = coeff; self.layer = layer; self.positions = positions
        self._handle = None

    def _hook_fn(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        steer = (self.coeff * self.vector).to(h.device, dtype=h.dtype)
        h = h.clone()
        if self.positions == "all":
            h += steer
        elif self.positions == "prompt":
            if h.shape[1] == 1: return output
            h += steer
        elif self.positions == "response":
            h[:, -1, :] += steer
        return (h,) + output[1:] if isinstance(output, tuple) else h

    def __enter__(self):
        self._handle = self.model.model.layers[self.layer].register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()

# Usage:
with ActivationSteerer(model, syc_vector, coeff=3.0, layer=STEER_LAYER, positions="all"):
    response = generate(model, tokenizer, question)
```

### Projection-based monitoring

The projection formula (from `eval/cal_projection.py` in the persona-vectors repo):

$$
\text{projection} = \frac{a \cdot v}{\|v\|}
$$

where $a$ is the mean response activation at the target layer and $v$ is the trait vector.

Implementation:
```python
def compute_trait_projections(model, tokenizer, system_prompts, questions, responses, trait_vector, layer):
    activations = extract_response_activations(model, tokenizer,
                    system_prompts, questions, responses, layer)   # (n, d_model)
    return ((activations @ trait_vector) / trait_vector.norm()).tolist()
```

Applied to three conditions on a held-out set of questions:
- **Baseline** (empty system prompt): projections cluster near zero.
- **Positive-prompted** (sycophantic system prompt): projections shift UP.
- **Steered** (coeff=3 during generation): projections shift even higher than positive-prompted.

Reading the plot: projection is a VALID monitor for trait expression - it correlates with autorater scores and separates the three conditions cleanly.

### Multi-trait pipeline: `run_trait_pipeline`

Refactored into a single function so we can run it for each trait:

```python
def run_trait_pipeline(model, tokenizer, trait_name, trait_data, layer=19,
                        steering_coefficients=None, max_new_tokens=256, override=False):
    """Full pipeline: generate -> score -> filter -> extract vector -> steer -> evaluate."""

    def load_or_generate(path, generate_fn, description):
        # Caches intermediate results to disk (json or .pt)
        path = Path(path)
        if path.exists() and not override:
            return json.loads(path.read_text()) if path.suffix == ".json" else t.load(path)
        result = generate_fn()
        # ... save ...
        return result

    responses = load_or_generate(f"{trait_name}_responses.json",
                    lambda: generate_contrastive_responses(model, tokenizer, trait_data, trait_name),
                    "responses")
    scored = load_or_generate(f"{trait_name}_scored.json",
                    lambda: score_all(responses, eval_prompt, model=AUTORATER_MODEL_GPT),
                    "scored responses")
    effective_pairs = filter_effective_pairs(scored, trait_data)
    trait_vectors = load_or_generate(f"{trait_name}_vectors.pt",
                    lambda: extract_contrastive_vectors(model, tokenizer, effective_pairs),
                    "trait vectors")
    steering_results = load_or_generate(f"{trait_name}_steering.json",
                    lambda: run_steering_experiment(model, tokenizer, trait_data["questions"],
                            trait_vectors[layer], trait_data["eval_prompt"], layer,
                            steering_coefficients),
                    "steering results")
    return trait_vectors, steering_results
```

Then `for trait in TRAITS: run_trait_pipeline(...)`.

### Multi-trait geometry

Compute cosine similarity between the 7 trait vectors at the target layer:

```python
trait_vectors_at_layer = t.stack([vecs[layer] for vecs in all_trait_vectors.values()])
cos_sim_matrix = t.nn.functional.cosine_similarity(
    trait_vectors_at_layer.unsqueeze(0), trait_vectors_at_layer.unsqueeze(1), dim=-1)
```

Findings:
- Most pairs have |cos_sim| < 0.5 - traits capture GENUINELY DIFFERENT behavioral dimensions.
- No two traits should have cos_sim > 0.8 (that would mean they're capturing the same thing).
- The MODEL'S BEHAVIORAL SPACE CAN'T BE CAPTURED BY A SINGLE AXIS (like the Assistant Axis) - multiple independent directions exist, each corresponding to a specific kind of behavioral shift.
- The Assistant Axis from Section 1 is probably a WEIGHTED COMBINATION of several of these trait directions.

---

## 5. Bonus - finetuning-induced persona shifts + preventative steering

### Bonus tasks (optional)
- **Finetuning-induced persona shift monitoring**: fine-tune the model on a dataset that contains subtly sycophantic examples. Monitor the sycophancy PROJECTION on held-out inputs BEFORE and AFTER finetuning. Verify that projection increases -> finetuning introduced a persona shift.
- **Preventative steering during finetuning**: apply steering during the training forward passes to PREVENT the model from acquiring the trait. Compared to naive finetuning + post-hoc steering, this gives cleaner separation and doesn't require intervention at inference time.

Not fully explored in the notebook, but included as extension exercises.

---

## Cross-cutting takeaways

- **Personas live as directions in activation space**. Whether we're talking about the global "how assistant-like" axis or per-trait vectors like sycophancy, the operation is always the same: extract mean activations under different conditions, subtract to isolate the direction of interest.
- **Mean over RESPONSE tokens**, not prompt tokens. Persona expression is a property of what the model GENERATES, not what it reads.
- **Middle-to-late layers** work best (roughly 60-80% of the way through). Early layers are too token-level; final layers are too logit-focused.
- **Contrastive prompting = persona vectors**. Positive/negative system prompts + autorater filtering + mean-difference = a clean trait vector.
- **Steering positions matter**. Steering at ALL POSITIONS during prefill modifies the KV cache and has a much stronger effect than last-token-only steering.
- **Activation capping is a soft intervention**. Only kicks in when projection exceeds a per-layer threshold. Preserves normal responses. Requires calibrated per-layer vectors + thresholds (a generic axis doesn't work).
- **Autorater filtering is essential** for contrastive prompting. Without filtering ineffective pairs, the trait vector gets diluted by no-op examples where the model ignored the polarity instruction.
- **Trait vectors are NOT a single direction**. Multi-trait cosine similarity shows |cos_sim| < 0.5 between most pairs. The model's behavioral space is multi-dimensional; the Assistant Axis is probably a weighted combination.
- **PC1 correlates strongly with the Assistant Axis** for a persona set spanning role-play to default. This suggests "assistant-ness" is a real, robust, first-order direction - not an artifact of persona selection.
- **Projection-based monitoring is passive**. Just project activations onto the trait vector to measure trait expression. No intervention needed. Ideal for detecting drift during long conversations.
- **Content filters and refusals**: Claude Haiku refuses to score "evil" and "hallucinating" traits. Fall back to GPT-4.1-mini for these.
- **PyTorch hooks + context managers**. All interventions in this chapter use `nn.Module.register_forward_hook` wrapped in `__enter__` / `__exit__` for guaranteed cleanup. Use this pattern for any activation intervention.
- **Chat template shenanigans**: Gemma 2 doesn't have a `"system"` role - merge system content into first user message. Qwen has a native system role. Always check the model's tokenizer's chat template before writing intervention code.
- **Two paper connection**: The Assistant Axis (single global axis) is a special case / weighted combination of the per-trait Persona Vectors. The latter is more surgical (measure/steer specific traits); the former is more robust as a drift-detector because it aggregates across many personas.
