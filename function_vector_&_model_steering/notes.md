# Function Vectors & Model Steering - Notes

Notes from ARENA [1.3.2] Function Vectors & Model Steering. Companion to [1_3_2_Function_Vectors_&_Model_Steering_exercises.ipynb](1_3_2_Function_Vectors_&_Model_Steering_exercises.ipynb).

The central question: **can a model be steered to produce different outputs by intervening on its forward pass with vectors found *without* gradient descent?** Two answers, both yes:

- **Function vectors** (Todd et al., [functions.baulab.info](https://functions.baulab.info/)) - average per-head outputs at the last position of an in-context-learning (ICL) prompt; the resulting vector, dropped into a *zero-shot* prompt's residual stream, induces ICL-task behavior with no exemplars.
- **Activation addition / steering vectors** (Turner et al., LessWrong 2023) - compute a residual-stream difference between two contrasting prompts, then add it (scaled) to a third prompt's residual stream at a chosen layer to steer generation.

Models used: GPT-J-6B (function vectors), GPT-2-XL (steering).

Tool: `nnsight` - a thin wrapper around HuggingFace models that exposes layers as Python objects whose `.input` / `.output` can be read or written inside a `with model.trace(...)` block. Interventions are written like regular tensor code; the library captures the AST and interleaves it with the forward pass.

---

## Big picture (the two paradigms)

- **Function vectors** are *task-encoded*. The h-vector is built by averaging hidden states (or attention-head outputs) at the **final token** of *successful* ICL prompts. Conceptually it answers: "what residual-stream state precedes the model emitting the correct answer for task T?" Adding it back to a zero-shot prompt at the right layer reproduces task-T behavior.
- **Steering vectors** are *concept-encoded*. The steering vector is built by running two contrasting prompts (e.g. `" Love"` vs `"Hate"`) and taking the **difference of activations** at some layer. Adding the scaled difference to a third prompt nudges its generation toward the positive side of the contrast.
- Both are inference-time, gradient-free, and **causal by construction** (the test is whether intervention changes output, not whether a probe classifies it).
- Mechanism differs:
  - Function vectors operate on the **mean of attention-head outputs** before they're summed and projected by `W_O`. The CIE metric attributes per-head contribution.
  - Steering vectors operate on the **residual stream directly**, no per-head decomposition.

---

## 1. Introduction to `nnsight`

### Tasks
- Loaded GPT-J-6B as `LanguageModel("EleutherAI/gpt-j-6b", device_map="auto", dtype=t.bfloat16)`. Module structure: `model.transformer.wte` (token embeddings), `model.transformer.h` (list of 28 transformer blocks), `model.lm_head` (unembedding).
- Learned the `nnsight` paradigm:
  - `with model.trace(prompt, remote=REMOTE):` opens a tracing context.
  - Inside, `model.transformer.h[layer].output[0]` etc. are *proxy objects*, not tensors. They're symbolic references to activations.
  - `.save()` marks a proxy to persist past the end of the trace. Without `.save()`, the value is discarded for memory.
  - Multiple `with tracer.invoke(prompts):` blocks inside one `trace()` run as **separate sequential forward passes inside the same context** - this is how cross-prompt interventions stay clean.
- Extracted layer-0 attention patterns from `model.transformer.h[0].attn.attn_dropout.input` (the softmaxed pattern just before dropout is applied) and visualized with `circuitsvis`.

```python
from nnsight import LanguageModel

model = LanguageModel("EleutherAI/gpt-j-6b", device_map="auto", dtype=t.bfloat16)
prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt, remote=REMOTE):
    # Proxy object. Without .save() it disappears at end of block.
    attn_patterns = model.transformer.h[0].attn.attn_dropout.input.save()

# After the context closes, attn_patterns is a real tensor.
attn_patterns_value = attn_patterns.squeeze(0)   # drop batch dim

str_tokens = [s.replace("Ġ", " ") for s in model.tokenizer.tokenize(prompt)]
display(cv.attention.attention_patterns(tokens=str_tokens, attention=attn_patterns_value))
```

### Learning
- The `model.trace(...)` block isn't lazy in the deferred-execution sense - it's *interleaved*. The intervention code runs while the model's forward pass runs. Anything not `.save()`d is garbage-collected at block exit.
- `.output` vs `.input`: many HuggingFace layers wrap their output in tuples (`output[0]` is the actual hidden state). Always check the type. For attention, the softmaxed pattern lives at `attn.attn_dropout.input` (before dropout consumes it).
- BPE tokens use `"Ġ"` to mark leading spaces (vs the prepended-space convention in TransformerLens). Strip and replace before display.
- `nnsight` works with both local and remote (NDIF) execution; the same code runs in both modes via `remote=REMOTE`.

---

## 2. Task-encoding hidden states (the h-vector)

The Function Vectors paper's first result: residual-stream activations at the last position of a successful ICL prompt encode the **task itself**. Adding the average back to a zero-shot prompt induces task-correct completions.

The running task is **antonyms**: prompts look like `"Q: hot\nA: cold\n\nQ: up\nA: down\n\nQ: tall\nA:"`. The completion should be the antonym of the final word.

### Tasks
- Set up `ICLSequence` and `ICLDataset` helper classes. Each prompt is `n_prepended` antonym demonstrations + a final query whose answer is held back. `ICLDataset` can also produce a **corrupted** variant where every non-final pair has its second word randomly shuffled - destroys the pattern but keeps the surface structure.
- Implemented `calculate_h`: average the residual-stream hidden state at the last token position across all ICL prompts, at a chosen layer. Use `model.transformer.h[layer].output[0][:, -1].mean(dim=0)`.
- Implemented `intervene_with_h`: run two forward passes in one trace, one clean and one where `h` is added to the residual stream at the last position. Compare argmax completions.
- Folded both into `calculate_h_and_intervene` (compute h *and* intervene in a single trace via three `invoke` blocks).
- Added a logprob-based variant `calculate_h_and_intervene_logprobs` to measure the change in correct-token log-probability (a finer-grained signal than argmax).

```python
def calculate_h(model, dataset, layer=-1):
    """Average final-position hidden state across all prompts at layer `layer`."""
    with model.trace(dataset.prompts):
        # GPT-2-style layers return tuples: .output[0] is the hidden state.
        hidden = model.transformer.h[layer].output[0]            # (batch, seq, d_model)
        h = hidden[:, -1].mean(dim=0).save()                     # (d_model,)
        # Bonus: also grab the model's natural next-token predictions.
        logits = model.lm_head.output[:, -1]
        next_tok = logits.argmax(dim=-1).save()
    completions = model.tokenizer.batch_decode(next_tok)
    return completions, h


def intervene_with_h(model, zero_shot_dataset, h, layer, remote=REMOTE):
    """Compare zero-shot completions with and without adding h to the residual stream."""
    with model.trace(remote=remote) as tracer:
        # Forward pass 1: zero-shot clean baseline.
        with tracer.invoke(zero_shot_dataset.prompts):
            tok_clean = model.lm_head.output[:, -1].argmax(dim=-1)
            completions_zero_shot = model.tokenizer.batch_decode(tok_clean).save()

        # Forward pass 2: same prompts, but add h at the final position of layer `layer`.
        with tracer.invoke(zero_shot_dataset.prompts):
            model.transformer.h[layer].output[0][:, -1] += h
            tok_steered = model.lm_head.output[:, -1].argmax(dim=-1)
            completions_intervention = model.tokenizer.batch_decode(tok_steered).save()

    return completions_zero_shot, completions_intervention


def calculate_h_and_intervene(model, dataset, zero_shot_dataset, layer):
    """All three forward passes (compute h, clean zero-shot, steered zero-shot) in one trace."""
    with model.trace() as tracer:
        # Invoke 1: compute h from clean ICL prompts.
        with tracer.invoke(dataset.prompts):
            h = model.transformer.h[layer].output[0][:, -1].mean(dim=0)

        # Invoke 2: clean zero-shot baseline.
        with tracer.invoke(zero_shot_dataset.prompts):
            clean_tok = model.lm_head.output[:, -1].argmax(dim=-1).save()

        # Invoke 3: zero-shot with h added at the final position.
        with tracer.invoke(zero_shot_dataset.prompts):
            model.transformer.h[layer].output[0][:, -1] += h
            steered_tok = model.lm_head.output[:, -1].argmax(dim=-1).save()

    return (tokenizer.batch_decode(clean_tok),
            tokenizer.batch_decode(steered_tok))


def calculate_h_and_intervene_logprobs(model, dataset, zero_shot_dataset, layer):
    """Same idea, but return correct-token logprobs (more sensitive than argmax)."""
    # First token of each correct completion (some answers tokenize to multiple tokens).
    correct_tokens = t.tensor(
        [ids[0] for ids in model.tokenizer(zero_shot_dataset.completions)["input_ids"]],
        device=model.device,
    )

    with model.trace() as tracer:
        with tracer.invoke(dataset.prompts):
            h = model.transformer.h[layer].output[0][:, -1].mean(dim=0)

        with tracer.invoke(zero_shot_dataset.prompts):
            logprobs = model.lm_head.output[:, -1].log_softmax(dim=-1)
            correct_lp_clean = logprobs[
                t.arange(len(zero_shot_dataset.prompts), device=model.device),
                correct_tokens,
            ].save()

        with tracer.invoke(zero_shot_dataset.prompts):
            model.transformer.h[layer].output[0][:, -1] += h
            logprobs_int = model.lm_head.output[:, -1].log_softmax(dim=-1)
            correct_lp_int = logprobs_int[
                t.arange(len(zero_shot_dataset.prompts), device=model.device),
                correct_tokens,
            ].save()

    return correct_lp_clean.tolist(), correct_lp_int.tolist()
```

### Results
- Zero-shot baseline on antonyms: ~0% correct (the model usually just copies the query word - the natural continuation when there's no in-context demonstration).
- Zero-shot + h-intervention at layer 12: ~25-50% correct completions (replicates the paper's finding on GPT-J).
- Logprob view: **even when argmax doesn't change, the logprob on the correct token consistently goes up** under intervention. The argmax view undersells the effect.
- Different random seeds for the ICL dataset and the zero-shot dataset are essential. Otherwise the intervention is being tested on prompts that were in the training set for h, which inflates the score.

### Learning
- The `nnsight` "multiple invokes per trace" pattern is the right way to do interventions cleanly. Each invoke is a separate forward pass but they share the same backing context, so a vector computed in invoke 1 can be used in invoke 3 without leaving the GPU.
- **`logits.log_softmax(-1)[arange(batch), correct_tokens]` is the per-example correct-token logprob lookup.** This is the same `gather`-style indexing used in cross-entropy.
- The h-intervention works at a *specific layer* (~12 of 28 for GPT-J-6B, mid-network). Earlier layers haven't formed the task representation; later layers are too close to the output to leave room for the model to "use" the injected vector.
- The corrupted-dataset trick (shuffle the answers in the demonstrations) is the standard way to construct a baseline that controls for surface form while removing the task signal.

---

## 3. Function vectors (per-head decomposition)

The paper's main result: not all of the residual stream encodes the task - **a small set of attention heads** are responsible. Identify them via Causal Indirect Effect (CIE), then sum their outputs to get a more concentrated "function vector."

### Tasks
- Implemented `calculate_fn_vectors_and_intervene`. For each (layer, head):
  1. **Clean run**: compute the mean per-head z-vector at the final position from clean ICL prompts. `out_proj.input` is the concatenated z (all heads) just before `W_O` mixes them, so reshape `[batch, d_model] -> [batch, n_heads, d_head]`, then mean over batch.
  2. **Corrupted baseline run**: shuffled prompts -> read off the correct-token logprob.
  3. **Intervention run**: corrupted prompts, but replace this one head's z at the final position with the clean mean. Read off the correct-token logprob.
  4. **CIE** = (intervened correct logprob) - (corrupted correct logprob), averaged over batch.
- All `n_layers * n_heads` interventions run in *one trace*, one invoke per (layer, head). Plotted as a heatmap; reproduces Figure 3(a) of the paper - a small bright cluster of high-CIE heads in mid layers.
- Implemented `calculate_fn_vector`: for a chosen `head_list = [(layer, head), ...]`, take each layer's clean mean z, zero out all heads not in the list, push the result through that layer's `out_proj`, and sum across layers. Final vector lives in residual-stream space.
- Multi-token generation: `model.generate(prompt, max_new_tokens=N)` returns a generator context with `generator.invoke(...)` blocks (one per prompt batch). The function vector is added at the last position of the residual stream during generation.

```python
def calculate_fn_vectors_and_intervene(model, dataset, layers=None):
    """For each (layer, head), CIE = correct-logprob(intervened) - correct-logprob(corrupted)."""
    LAYERS = list(range(model.config.n_layer)) if layers is None else list(layers)
    n_heads = model.config.n_head
    d_head = model.config.n_embd // model.config.n_head
    batch_size = len(dataset)
    corrupted = dataset.create_corrupted_dataset()

    correct_ids = t.tensor(
        [toks[0] for toks in model.tokenizer(dataset.completions)["input_ids"]],
        device=model.device,
    )

    z_dict = {}                   # (layer, head) -> mean clean z at final pos, shape (d_head,)
    correct_logprobs_dict = {}    # (layer, head) -> intervened logprob, shape (batch,)

    with model.trace(remote=REMOTE) as tracer:
        # Clean run: store mean per-head z at final pos for every layer.
        with tracer.invoke(dataset.prompts):
            for layer in LAYERS:
                # out_proj.input is the concat of all heads' z's before W_O.
                z = model.transformer.h[layer].attn.out_proj.input[:, -1]    # (b, d_model)
                z = z.reshape(batch_size, n_heads, d_head)
                z_head_mean = z.mean(dim=0)                                   # (h, d_head)
                for head in range(n_heads):
                    z_dict[(layer, head)] = z_head_mean[head]

        # Corrupted baseline: just read off correct-token logprob.
        with tracer.invoke(corrupted.prompts):
            logprobs = model.lm_head.output[:, -1].log_softmax(dim=-1)
            correct_lp_corrupted = logprobs[
                t.arange(batch_size, device=model.device), correct_ids,
            ].save()

        # One intervention per (layer, head): patch this head's final-pos z with clean mean.
        for layer in LAYERS:
            for head in range(n_heads):
                with tracer.invoke(corrupted.prompts):
                    z = model.transformer.h[layer].attn.out_proj.input[:, -1]
                    z.reshape(batch_size, n_heads, d_head)[:, head, :] = z_dict[(layer, head)]
                    logprobs = model.lm_head.output[:, -1].log_softmax(dim=-1)
                    correct_logprobs_dict[(layer, head)] = logprobs[
                        t.arange(batch_size, device=model.device), correct_ids,
                    ].save()

    # Stack into (layers, heads, batch), subtract baseline, average over batch.
    stacked = t.stack([correct_logprobs_dict[(l, h)]
                       for l in LAYERS for h in range(n_heads)])
    stacked = einops.rearrange(stacked, "(L H) B -> L H B", L=len(LAYERS), H=n_heads)
    return (stacked - correct_lp_corrupted).mean(dim=-1)        # (layers, heads)


def calculate_fn_vector(model, dataset, head_list):
    """Sum of selected heads' contributions to the residual stream, averaged over prompts.

    Trick: zero out the non-selected heads' z's, then pass through W_O. This gives the
    *exact* vector those heads would have written to the residual stream.
    """
    # Group by layer.
    head_dict = defaultdict(set)
    for layer, head in head_list:
        head_dict[layer].add(head)

    fn_vector_list = []
    with model.trace(dataset.prompts):
        for layer, selected_heads in head_dict.items():
            out_proj = model.transformer.h[layer].attn.out_proj
            # Mean concatenated-z at final pos, averaged over prompts.
            hidden_states = out_proj.input[:, -1].mean(dim=0)            # (d_model,)
            # Zero every head not in our selection.
            for head in set(range(N_HEADS)) - selected_heads:
                hidden_states.reshape(N_HEADS, D_HEAD)[head] = 0.0
            # Push through out_proj to get the residual-stream contribution.
            out = out_proj(hidden_states.unsqueeze(0)).squeeze()         # (d_model,)
            fn_vector_list.append(out.save())

    return sum(v for v in fn_vector_list)


def intervene_with_fn_vector(model, word, layer, fn_vector,
                             prompt_template='The word "{x}" means', n_tokens=5):
    """Generate two continuations: one normal, one with fn_vector added at the last position."""
    prompt = prompt_template.format(x=word)
    with model.generate(max_new_tokens=n_tokens, remote=REMOTE) as gen:
        with gen.invoke(prompt):
            unsteered_out = model.generator.output.save()
        with gen.invoke(prompt):
            # Add at every generation step at the last residual position.
            model.transformer.h[layer].output[0][:, -1] += fn_vector
            steered_out = model.generator.output.save()
    return (model.tokenizer.batch_decode(unsteered_out)[0],
            model.tokenizer.batch_decode(steered_out)[0])
```

### Results
- CIE heatmap on antonyms (GPT-J-6B): a clear bright cluster around layers 8-16, with ~10-12 individual heads dominating. The selected heads (`(8,0), (8,1), (9,14), (11,0), (12,10), (13,12), (13,13), (14,9), (15,5), (16,14)` etc.) reproduce the paper's set.
- Function vector intervention with prompt `'The word "light" means'` (and `"light"` removed from the antonym pool first):
  - Unsteered: model defines `"light"` normally ("not heavy", "illumination", ...).
  - Steered: model produces antonyms like `"dark"`, `"heavy"`. The intervention generalizes to a word held out from the dataset used to build the FV.
- Same selected heads transfer to a different task (country-capital pairs) with reduced but still real effect. The heads are not purely antonym-specific - they encode a more general "ICL-task-completion" computation.

### Learning
- **`out_proj.input` is the cleanest single point to read per-head z vectors** in a HuggingFace model. The library stores Q, K, V projections separately, but the post-attention z is concatenated. Reshape `[batch, d_model] -> [batch, n_heads, d_head]` to recover per-head views.
- **CIE (Causal Indirect Effect)** is the right metric for per-head attribution: it's directly causal (we changed something and measured the output change) and additive in the right way (sums to roughly the total effect when heads are independent). Compare to logit attribution, which is correlational.
- **Run all interventions inside one trace.** Spawning a separate forward pass per (layer, head) would be 12 * 28 = 336 forward passes; one trace with 336 invokes is one batched forward pass with the same compute.
- **Function vector vs h-vector**: the FV is more concentrated (sum of ~10 specific head outputs vs the mean of the entire residual stream) and produces stronger / more reliable steering. The h-vector is a useful first sanity check; the FV is the production version.
- **Multi-token generation needs `model.generate(...)`**, not `model.trace(...)`. Inside, `generator.invoke(...)` creates one forward chain per prompt batch, and the intervention is applied at every generation step that runs inside the block.

---

## 4. Steering vectors (GPT-2-XL, activation addition)

Different mechanism, same spirit. Compute a steering vector by **subtracting one prompt's residual-stream activations from another's**, then add it (scaled) to a third prompt during generation.

### Tasks
- Loaded GPT-2-XL (`gpt2-xl`) locally (NDIF doesn't host it). Used GPT-2-XL's 48 layers and 1600-dim residual stream.
- Implemented `calculate_and_apply_steering_vector`. The API takes:
  - `prompt`: the prompt to steer.
  - `activation_additions`: list of `(layer, coeff, steering_prompt)` tuples. Each contributes its residual-stream activations at `layer` (taken from `steering_prompt`), multiplied by `coeff`, added to the first `seq_len` positions of `prompt`'s forward pass.
- Used `model.generate(...)` with `do_sample=True, top_p=0.3, repetition_penalty=1.2`. Compared `n_comparisons` unsteered samples vs `n_comparisons` steered samples for the same prompt.

```python
SAMPLING_KWARGS = {"do_sample": True, "top_p": 0.3, "repetition_penalty": 1.2}


def calculate_and_apply_steering_vector(
    model, prompt, activation_additions, n_tokens, n_comparisons=1, use_bos=True,
):
    """Each (layer, coeff, steer_prompt) in activation_additions contributes its layer-`layer`
    activations (scaled by coeff) to the first seq_len positions of `prompt`'s generation."""

    act_add_prompts = [a[2] for a in activation_additions]
    act_add_layers  = [a[0] for a in activation_additions]
    act_add_coeffs  = [a[1] for a in activation_additions]
    act_add_seq_lens = [len(model.tokenizer.tokenize(p)) for p in act_add_prompts]
    assert len(set(act_add_seq_lens)) == 1, "All steering prompts must be the same length."
    assert act_add_seq_lens[0] <= len(model.tokenizer.tokenize(prompt))

    unsteered_prompts = [prompt for _ in range(n_comparisons)]
    steered_prompts   = [prompt for _ in range(n_comparisons)]

    with model.generate(max_new_tokens=n_tokens, remote=REMOTE, **SAMPLING_KWARGS) as gen:
        # Pass 1: extract per-steering-prompt activations at the requested layers.
        with gen.invoke(act_add_prompts):
            act_add_vectors = [
                model.transformer.h[layer].output[0][i, -seq_len:]
                for i, (layer, seq_len) in enumerate(zip(act_add_layers, act_add_seq_lens))
            ]

        # Pass 2: unsteered baseline.
        with gen.invoke(unsteered_prompts):
            unsteered_out = model.generator.output.save()

        # Pass 3: add scaled steering vectors to the residual stream of the real prompt.
        with gen.invoke(steered_prompts):
            for i, (layer, coeff, seq_len) in enumerate(
                zip(act_add_layers, act_add_coeffs, act_add_seq_lens)
            ):
                model.transformer.h[layer].output[0][:, :seq_len] += coeff * act_add_vectors[i]
            steered_out = model.generator.output.save()

    return (model.tokenizer.batch_decode(unsteered_out[-n_comparisons:]),
            model.tokenizer.batch_decode(steered_out[-n_comparisons:]))
```

### Results
Replicated four of the post's examples on GPT-2-XL:

| Prompt | Activation additions | Effect |
|---|---|---|
| `"I hate you because"` | `(6, +8.0, " Love")`, `(6, -8.0, "Hate")` | Continuations turn affectionate ("I love you because you're so kind...") instead of hostile. |
| `"This person is"` | `(6, +8.0, " Love")`, `(6, -8.0, "Hate")` | Continuations consistently describe the person positively. |
| `"I went up to my friend and said"` | `(20, +4.0, "I talk about weddings constantly  ")`, `(20, -4.0, "I do not talk about weddings constantly")` | Steered completions mention weddings, vows, the bride, even when irrelevant. Most robust of the examples. |
| `"To see the eiffel tower, people flock to"` | `(24, +10.0, "The Eiffel Tower is in Rome")`, `(24, -10.0, "The Eiffel Tower is in France")` | Steered completions place the Eiffel Tower in Italy/Rome with surprising consistency. |

### Learning
- **Activation addition is shockingly cheap.** No training, no gradient steps, just two forward passes' worth of residual-stream activations subtracted and added back. The simplicity is the headline result.
- **The contrast pair matters.** `(" Love", "Hate")` works because both are short, single-token-ish, and tokenize to comparable sequences. Including a leading space on one side and not the other is a common gotcha (the post's original examples used this pattern intentionally).
- **Layer choice is everything.** Earlier layers (6) work for emotional/conceptual steering; later layers (20-24) work for factual / topic steering. The post explores this; the exercises just use the values that work.
- **Coefficient is a tuning knob between effectiveness and coherence.** Too low and the steering doesn't show; too high and the output degenerates into gibberish ("the love love love love..."). Typical range: 4-10 for these examples.
- **Sequence-length matching.** The activation-addition prompts must be the same length (so they can be added to the same residual positions). Pick contrast pairs that tokenize identically when possible, or pad / truncate.

---

## Takeaways

1. **Interventions reveal what classifiers can't.** A probe shows a representation is *correlated* with task T. Function vectors / steering vectors *cause* task-T behavior. The difference is exactly the difference between observation and intervention.
2. **`nnsight`'s trace/invoke model maps cleanly onto causal experiments.** One trace = one batched forward pass with multiple sequential sub-passes that share state. This is the right primitive for "compute X from prompts A, then patch X into prompts B."
3. **Final-position activations dominate.** Both h-vector and function-vector use the **last token's** activations exclusively, because that's where the model's answer is being formed. Other positions matter for some interventions (steering across full prompts) but not for task encoding.
4. **`out_proj.input` is the per-head z access point** in HuggingFace models. Reshape to `[batch, n_heads, d_head]`. There's no per-head buffer the library exposes natively, but this reshape is exact.
5. **CIE > observation.** Per-head attribution by intervention (replace one head's z with a clean version inside a corrupted run) gives a calibrated, additive metric. Use this rather than reading attention patterns when looking for task-relevant heads.
6. **Mid-network layers are where the task representation lives.** For h-vector (layer 12 of 28 in GPT-J), for steering (layers 6-24 of 48 in GPT-2-XL). Too early = features haven't formed; too late = no room to use the injection.
7. **Activation addition generalizes the function-vector idea.** Both are residual-stream additions; both are gradient-free; both work because the residual stream is a privileged additive bus. The difference is just how the added vector is constructed (contrast of prompts vs mean of head outputs).

---

## Mini-glossary

- **`nnsight` `LanguageModel`**: a HuggingFace model wrapped so that submodule attributes (`.input`, `.output`) become proxies inside a `with model.trace(...)` block.
- **`.save()`**: marks a proxy to persist past the end of the trace; unmarked proxies are GC'd to save memory.
- **`tracer.invoke(prompts)`**: a sub-pass within one trace. Multiple invokes run sequentially, share state, batch together at the GPU level.
- **`model.generate(...)` / `generator.invoke(...)`**: the multi-token-generation analog of trace/invoke. Interventions inside fire at every generation step within the invoke.
- **h-vector**: mean residual-stream hidden state at the final position of successful ICL prompts. Encodes the task.
- **Function vector (FV)**: sum across selected (layer, head) pairs of the residual-stream contribution from that head's mean final-position z. More concentrated than h-vector.
- **CIE (Causal Indirect Effect)**: `correct_logprob(intervened) - correct_logprob(corrupted_baseline)`, averaged over examples. Per-head causal attribution metric.
- **Corrupted ICL dataset**: same prompt structure, but each non-final demonstration's answer is randomly shuffled. Surface form preserved, task signal destroyed.
- **`out_proj` (HuggingFace)**: the linear layer that combines `W_O @ z` for all heads in a transformer attention block. `out_proj.input` is the concatenated z; reshape `[batch, d_model] -> [batch, n_heads, d_head]` for per-head access.
- **Activation addition / steering vector**: `coeff * (act(prompt_pos) - act(prompt_neg))`, added to the residual stream of a third prompt at a chosen layer. Turner et al.'s approach.
- **`model.generator.output`**: the generated token IDs from `model.generate(...)`. Decode with `tokenizer.batch_decode`.
