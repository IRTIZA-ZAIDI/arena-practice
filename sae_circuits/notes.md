# SAE Circuits - Notes

Notes from ARENA [1.4.2] SAE Circuits. Companion to [1_4_2_SAE_Circuits_exercises.ipynb](1_4_2_SAE_Circuits_exercises.ipynb).

The central question: SAEs from [1.3.3](../interpretability_with_SAEs/notes.md) gave a monosemantic feature basis at a single site. Can those features be assembled into **circuits** across layers - showing how latents in one layer causally drive latents in the next, and ultimately drive logits? Four passes at the problem, each more powerful than the last:

1. **Latent gradients** - Jacobian-based edges between active latents at two layers, using functorch's `jacrev`.
2. **Transcoders** - MLP replacements that learn the *computation*, not just the reconstruction. Enable "pullback" and "de-embedding" analyses that don't require a specific input.
3. **Attribution graphs** - the linearized-model + gradient-attribution pipeline from Anthropic's Circuit Tracing paper. Frozen attention + LayerNorm + skip-connection MLPs let residual-stream flow be traced exactly.
4. **Circuit-tracer library + interventions** - the same pipeline packaged as a library, plus supernode-level ablations on the Dallas/Austin two-hop factual-recall circuit. Companion cheat sheet: [circuit-tracer.md](circuit-tracer.md).

Models: GPT-2-small (sections 1-2) and **Gemma-3-1B-IT** with **GemmaScope 2 transcoders** (section 3), then **Gemma-2-2B** via the pre-built ReplacementModel in circuit-tracer (section 4). Everything gradient-based lives inside a `HookedSAETransformer` context so SAEs/transcoders can be attached and detached without touching the base model.

---

## Big picture (three moves from single-latent to circuit-level)

**Move 1: pairwise gradients between active latents.** Given two SAEs at layers `L1 < L2`, run the model on a prompt, capture the active latents at both sites, and compute the Jacobian of `L2`'s latent activations with respect to `L1`'s. Result: a `(active_L1, active_L2)` matrix where entry `(i, j)` says "how much does latent `i` at layer `L1` contribute to latent `j` at layer `L2` on this input?" Cheap and honest, but **only covers currently-active latents on this specific prompt**.

**Move 2: transcoders + input-free analysis.** Standard MLP-SAEs reconstruct the *output* of an MLP. Transcoders learn a sparse decomposition of the MLP's *input->output map*: `y = ReLU(x @ W_enc) @ W_dec` approximates `MLP(x)` directly. Because the transcoder is a *replacement*, its encoder/decoder vectors both live in `d_model` space, so the dot product `sae_lower.W_dec[i] @ sae_upper.W_enc[:, j]` is a **pullback** that estimates "how much does latent `i` write into a direction that latent `j` reads" - **without any specific input**. Combined with the extended embedding (`W_E` mapped through layer-0 MLP transcoder) it gives an input-free "de-embedding" showing what tokens most activate a latent.

**Move 3: linearize the model globally.** Attention softmax, LayerNorm scale, and MLP nonlinearity are the only three non-linearities in a transformer. Freeze the softmax and LayerNorm scales at their reference-pass values, replace each MLP with a linear skip connection through the transcoder decoder, and every residual-stream path becomes linear. Now gradient attribution is *exact* rather than a local approximation: pick a target logit direction, backpropagate, and read off the influence of every earlier latent / embedding / MLP-error term.

---

## 1. Latent gradients

The simplest circuit tool: for two SAEs at layers `L_from` and `L_to`, compute the Jacobian of the upper latent activations w.r.t. the lower ones. Uses `torch.func.jacrev` on a sparse representation to keep it cheap.

### Tasks
- Loaded a stack of residual-stream SAEs across every GPT-2-small layer:
  ```python
  gpt2 = HookedSAETransformer.from_pretrained("gpt2-small")
  gpt2_saes = {
      layer: SAE.from_pretrained("gpt2-small-res-jb", f"blocks.{layer}.hook_resid_pre",
                                 device=device)
      for layer in range(gpt2.cfg.n_layers)
  }
  ```
- Built a `SparseTensor` helper that stores an activation tensor in two forms simultaneously - the dense `(n, d_sae)` version and a sparse `(values, indices, shape)` triple - so functorch's `jacrev` can differentiate through the sparse values only (would be 24k x 24k dense and pointless).
- Implemented `latent_acts_to_later_latent_acts`: takes lower-SAE latents (sparse form) -> decode to residual stream -> run the transformer's layers between `L_from` and `L_to` -> encode into upper-SAE latents. This is the function to Jacobian-through.
- Implemented `latent_to_latent_gradients` using `t.func.jacrev(fn, argnums=0, has_aux=True)`. The `has_aux=True` pattern is essential - the function returns `(main_output, aux)` and `jacrev` differentiates only the main output while still returning the aux. This is how the upper-SAE latent activations *and* the Jacobian come out of one pass.
- Same recipe with different function signatures:
  - **Token-to-latent** (`tokens_to_latent_acts`): differentiate through per-token embedding-scale factors instead of lower-SAE latents. Shows which input tokens activate a given latent.
  - **Latent-to-logit** (`latent_acts_to_logits`): differentiate through lower-SAE latents but output logits (or a subset of tokens via `token_ids`). Shows which latents most affect a chosen output token.

```python
from torch.func import jacrev

def latent_acts_to_later_latent_acts(
    latent_acts_nonzero,      # (n_nonzero,)         active values only
    latent_acts_indices,      # (n_nonzero, ndims)   where in the dense tensor
    latent_acts_shape,        # tuple                original dense shape
    sae_from, sae_to, model,
):
    # 1. Rebuild dense lower-latent tensor from sparse triple.
    latents = t.zeros(latent_acts_shape, device=latent_acts_nonzero.device)
    latents[tuple(latent_acts_indices.T)] = latent_acts_nonzero
    # 2. Decode into residual stream at layer L_from.
    resid = latents @ sae_from.W_dec + sae_from.b_dec
    # 3. Run the intermediate transformer layers.
    resid = model.run_between_hooks(
        resid, sae_from.cfg.hook_name, sae_to.cfg.hook_name,
    )
    # 4. Encode into upper-SAE latents.
    upper_latents = t.relu((resid - sae_to.b_dec) @ sae_to.W_enc + sae_to.b_enc)
    return upper_latents, (upper_latents,)     # (main, aux) for has_aux


def latent_to_latent_gradients(tokens, sae_from, sae_to, model):
    # Cache actual latent acts at both sites (dense).
    _, cache = model.run_with_cache_with_saes(
        tokens, saes=[sae_from, sae_to],
        names_filter=lambda n: n.endswith("hook_sae_acts_post"),
    )
    lower_dense = cache[f"{sae_from.cfg.hook_name}.hook_sae_acts_post"]
    lower_sparse = SparseTensor.from_dense(lower_dense)

    jac_fn = jacrev(latent_acts_to_later_latent_acts, argnums=0, has_aux=True)
    grads, (upper_dense_recon,) = jac_fn(
        lower_sparse.values, lower_sparse.indices, lower_sparse.shape,
        sae_from, sae_to, model,
    )
    # grads: (n_active_upper, n_active_lower) Jacobian.
    return grads, lower_sparse, SparseTensor.from_dense(upper_dense_recon), ...
```

### Results
- On `"The Eiffel tower is in Paris"` across layers 0 -> 3: the gradient matrix is sparse. Latents fire in the low tens per position; only a handful of `(lower, upper)` pairs have nonzero gradient.
- One clear pair: `(L0.F16911 at " E") -> (L3.F15266 at "iff")` - a "words that start with E" bigram-like feature composing with a token-continuation feature.
- Token-to-latent gradients show which input tokens drive a given latent (e.g. the Paris-token feature at L9 is driven by the "Eiffel" and "tower" positions).
- Latent-to-logit gradients on `"The Eiffel tower is in the city of"` with answer `" Paris"` surface a small set of L9 latents that directly boost the Paris logit.

### Learning
- **`has_aux=True` is the trick** for computing "the Jacobian AND the function output" without running the model twice. Standard `jacrev(fn)` only returns the Jacobian.
- The sparse pattern (`SparseTensor.from_dense`) avoids materializing a `(24k, 24k)` Jacobian - only the currently-active latents are differentiated.
- **Latent gradients are per-input.** A gradient of zero doesn't mean two latents can never interact - it means they don't happen to both be active on this prompt. This is the fundamental limitation that transcoders + pullbacks in Section 2 are designed to fix.

---

## 2. Transcoders

An MLP transcoder replaces the whole MLP block: `MLP(x) ~ ReLU(x @ W_enc + b_enc) @ W_dec + b_out`. Both encoder and decoder vectors live in **residual-stream space**, so cross-layer connections can be computed as pure weight products - **input-independent**.

### Tasks
- Unloaded GPT-2 SAEs to free memory, then loaded a stack of pretrained transcoders at every layer:
  ```python
  gpt2_transcoders = {
      layer: SAE.from_pretrained(
          release="callummcdougall/arena-demos-transcoder",
          sae_id=f"gpt2-small-layer-{layer}-mlp-transcoder-folded-b_dec_out",
          device=device,
      )
      for layer in range(gpt2.cfg.n_layers)
  }
  ```
- Wrote `run_with_cache_with_transcoder`: uses SAELens's transcoder support to attach and cache. With `use_error_term=True`, the model's real MLP output flows through downstream (lossless); the transcoder's latent activations are still recorded.
- Studied one specific latent (`layer 8, F1`) using two views:
  - **Top logits** (`W_dec[F1] @ W_U`): what tokens does F1 push for / against in the output?
  - **De-embedding** (`W_E @ W_enc[:, F1]`): what tokens most activate F1 based on their embedding?
- Discovered the **tied-embedding trap** in GPT-2: `W_E = W_U.T`, so top de-embedding tokens look almost identical to top logit tokens - because a token's embedding *is* the direction the model uses to score it at the output. A latent that reads from "Barack" also has the "Barack" direction show up in its unembed-side dot product, even if it has nothing to do with the concept of Barack.
- Implemented `create_extended_embedding`: pass `W_E` through layer-0 LayerNorm + layer-0 MLP transcoder + skip connection to get an "extended embedding" that avoids the tied-weight artifact. Divide by std for scale normalization. Recomputed de-embeddings using this: much cleaner semantic clusters.
- **Blind case study** on `layer 8, F479`: given only the latent index and no prior information, identify what it does by chaining four analyses:
  1. De-embedding: what tokens activate it?
  2. Logit lens: what does it push at the output?
  3. Pullback from earlier transcoder latents (`W_dec[earlier] . W_enc[:, F479]`) - which earlier latents write into F479's read direction?
  4. Component attribution via attention: which upstream heads route information that flows into F479?

```python
# Standard SAE dashboard: logit lens.
logits = sae.W_dec[latent_idx] @ model.W_U         # (d_vocab,)
top_tokens = logits.topk(10).indices

# De-embedding for a transcoder latent (residual-space encoder).
de_emb = model.W_E @ sae.W_enc[:, latent_idx]      # (d_vocab,)
# BUT: tied embeddings pollute this. Use the extended version instead.


def create_extended_embedding(model):
    """Pass W_E through the layer-0 MLP transcoder to get a cleaner embedding."""
    x = model.W_E                                             # (d_vocab, d_model)
    x = x / x.std(dim=-1, keepdim=True)                       # LN-ish rescale
    # Skip + transcoder output at layer 0.
    tc = gpt2_transcoders[0][0]
    x = x + t.relu(x @ tc.W_enc + tc.b_enc) @ tc.W_dec
    return x / x.std(dim=-1, keepdim=True)


ext_emb = create_extended_embedding(gpt2)
de_emb_clean = ext_emb @ sae.W_enc[:, latent_idx]


# Cross-layer pullback: which earlier transcoder latents write into the target's read direction?
target_read = target_transcoder.W_enc[:, target_latent]      # (d_model,)
earlier_writes = earlier_transcoder.W_dec                    # (n_latents, d_model)
pullback = earlier_writes @ target_read                      # (n_latents,)
# Top-k pullback = candidate earlier latents feeding into target.
```

### Results
- The blind-study latent (`8.F479`) turns out to be a "hesitation / speech disfluency" feature - fires on tokens like `" um"`, `" uh"`, `" like"`, and boosts tokens that continue those patterns. All four analyses (de-embedding, logit lens, pullback, attention attribution) converge on this reading.
- Pullback identifies specific earlier latents that write into `F479` (usually in layers 5-7) - visible even though not every earlier latent is active on every input.
- Extended embedding cleans up de-embedding results substantially: many latents that looked like "reads from token X" via naive de-embedding turn out to be reading from *concept* X after passing through layer-0 MLP.

### Learning
- **Transcoders are decoders that live in `d_model` space at both ends.** That's the entire reason pullbacks work - two transcoder vectors from different layers can be dotted without a forward pass. Standard MLP-SAEs put the encoder in MLP-hidden space, which doesn't compose.
- **Tied embeddings are a silent trap.** GPT-2's `W_E = W_U.T` means `de_embedding` and `top_logits` look nearly identical. The extended embedding (`W_E + MLP0(W_E)`) breaks the identity and reveals the semantic clustering that was hiding.
- **Four converging analyses beat any one alone.** The blind case study is a template: de-embedding tells what activates the latent, logit lens tells what it writes for, pullback identifies upstream sources, attention attribution identifies upstream routes. Any one alone is ambiguous; the conjunction is nearly unambiguous.

---

## 3. Attribution graphs (linearized-model gradient attribution)

The main technical exercise of the notebook. Build a full attribution graph over transcoder latents, embeddings, MLP errors, and output logits - with **linearized flow through the model** so every edge weight is exact rather than a local approximation.

> Visual companion: [sae_circuits.excalidraw](sae_circuits.excalidraw) - a full-notebook diagram with 7 sections covering latent gradients, transcoders, the local replacement model, reading/writing vector abstraction, the attribution graph in grid form, the end-to-end pipeline, and interventions. Math intuition boxes (∂) and prerequisite callouts (★) inline.

### Tasks

**Setup**: loaded Gemma-3-1B-IT (using HF gated access via `HF_TOKEN`) and the GemmaScope 2 transcoder stack for every layer. GemmaScope 2 provides 8 transcoders per layer (2 widths x 2 sparsities x 2 affine variants); used `width=16k, l0=small, affine=True, instruction_tuned=True`.

**Linearizing the model**: two helper classes.
- `FreezeHooks` (provided): caches attention patterns and LayerNorm scales on a reference forward pass, then installs TransformerLens forward hooks that replace them with the cached values on subsequent runs. With attention patterns fixed and LN scales fixed, attention becomes a linear function of values (which are linear in the residual stream), and LayerNorm becomes a linear rescale.
- `TranscoderReplacementHooks` (implemented as an exercise): for each layer, installs hooks that (a) capture the transcoder's latent activations on the forward pass, (b) block the real MLP gradient on the backward pass and route it through the transcoder's linear skip connection instead. The skip trick keeps the forward pass exact (real MLP outputs flow downstream) while making the backward pass linear.

**Sanity check**: verified average L0 per transcoder layer stays under 50 (typically 10-20). Higher L0 would explain any unexpected latent contributions as noise rather than real circuit structure.

**Building the graph**: three node types plus the output.
- `EMBEDDING` nodes: one per token position; writing vector = token embedding, no reading vector (they're sources).
- `LATENT` nodes: top-k transcoder latents per (layer, position); writing = `act * W_dec[latent]`, reading = `W_enc[:, latent]`.
- `MLP_ERROR` nodes: one per (layer, position); writing = `real_mlp_out - transcoder_reconstruction` (the residual the transcoder failed to explain), no reading vector.
- `LOGIT` nodes: one per top predicted next-token; reading vector = `demeaned W_U[:, tok]` (i.e. `W_U[:, tok] - W_U.mean(dim=-1)` to focus on token-specific evidence, not the average).

```python
# Salient logits at the final position.
def compute_salient_logits(model, logits, n_output_nodes=3):
    probs = logits[0, -1].softmax(dim=-1)
    top_tokens = probs.topk(n_output_nodes).indices
    W_U_centered = model.W_U - model.W_U.mean(dim=-1, keepdim=True)
    reading_vecs = W_U_centered.T[top_tokens]                 # (n_output, d_model)
    top_info = [(model.tokenizer.decode(t), probs[t].item()) for t in top_tokens]
    return reading_vecs, top_info


# The reading/writing vector abstraction.
# Every node has:
#   writing_vec: the residual-stream direction it ADDS when it fires
#   reading_vec: the residual-stream direction it READS to determine firing
# An edge (source -> target) is source_write . target_read, integrated through
# the linearized model between them.
```

**Computing edges**: for each target node, install a gradient seed at its (layer, position) that projects the residual stream onto its reading vector. Backpropagate through the frozen + transcoder-replaced model. The gradient at any earlier residual-stream position dotted with a source node's writing vector = the edge weight from source to target. Batched across target nodes for efficiency.

**Normalize + influence + prune**:
- Row-normalize the adjacency matrix by `|A|` sums so each target's incoming edges sum to 1.
- **Influence** via a truncated Neumann series: `I = (I + A + A^2 + ...) @ logit_weights` where the sum terminates at `n_layers` iterations because the graph is strictly lower-triangular (edges only go early -> late). Each node's influence = its total direct + indirect contribution to the top logits.
- **Prune** nodes below an influence threshold. Then prune edges below a threshold-of-max within each target.

**Visualize**: `utils.create_attribution_dashboard(result, model)` produces an HTML dashboard with layered node arrangement, edge weights, and per-latent activation histograms (populated by downloading Gemma Scope's pre-computed example data for every layer).

### Results
- The dashboard shows a clear layered structure: embeddings at the bottom, then transcoder latents forming semantic clusters (e.g. "capital city" tokens, "state" tokens), edges pointing upward through mid layers, and top-logit nodes at the output.
- Adjacency matrix is sparse after pruning - most (source, target) pairs have essentially zero influence. The interesting structure is a small subset.
- The skip-connection trick keeps forward-pass logits identical to the base model (checked: `max(|logits_with_hooks - logits_original|) < 1e-4`), so backward-pass attribution is measuring the *actual* model, not a distorted linearization of it.

### Learning
- **Linearizing while preserving the forward pass** is the key trick. Naive linearization (replace MLPs with a linear approximation everywhere) would degrade model behavior; the skip-connection hook keeps `mlp_out` intact for downstream layers while routing backward gradients through the linear surrogate.
- **The reading/writing vector abstraction is the unifying frame**. Every node type maps into "what direction does this write into the residual stream" and "what direction does it read". Once that frame is committed to, edges are just `source_write . linearized_flow . target_read` dot products.
- **Neumann series works because the graph is layered.** An `n_layers`-step matrix power converges exactly (higher powers are zero) because information only flows from early layers to late.
- **MLP-error nodes are essential**. Without them, the transcoder's reconstruction gap looks like unexplained influence flowing from nowhere. Making the error explicit lets the graph honestly represent "the transcoder didn't capture this piece."
- **Demean the logit reading vectors.** `W_U[:, tok]` on its own leaks the average unembed direction into every output edge; subtracting the mean makes the reading vector focus on what's specific about that particular token.

---

## 4. Exploring circuits and interventions with `circuit-tracer`

The `circuit-tracer` library (see the [tutorial](circuit-tracer.md) for API details) wraps the pipeline from Section 3 - loading models with transcoders, building attribution graphs, running interventions - behind a simpler API. This section uses it to explore the **Dallas/Austin two-hop factual-recall circuit**.

### The circuit

Prompt: `"Fact: the capital of the state containing Dallas is"`. Correct completion: `" Austin"`.

Two-hop reasoning:
1. Dallas -> Texas (which state?)
2. Texas + "capital of" -> Austin

Explored the pre-built attribution graph on Neuronpedia (`gemma-fact-dallas-austin`). Supernodes identified:
- `capital`, `state`: read from the prompt template.
- `Dallas`: reads the entity token.
- `Texas`: the intermediate "which state" representation.
- `Say a capital`: general "output should be a city" signal.
- `Say Austin`: the specific output latent.

### Tasks

**Loading**: `ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="transformerlens")` loads Gemma-2-2B with all the transcoders + freeze machinery pre-attached. `circuit_tracer_attribute(prompt, model)` runs the full pipeline from Section 3.

**Supernode construction**: extracted supernode feature lists from a Neuronpedia URL via `utils.extract_supernode_features(url)`. Built `Supernode` objects wrapping lists of `Feature(layer, pos, feature_idx)`, then arranged them into an `InterventionGraph` with layered `ordered_nodes` (embeddings at the bottom, output features at the top).

**Predict-then-test ablations**: wrote down predicted effects for each supernode ablation *before* running the code. Then ran each and compared:
- Ablate "Say a capital" (`-2x`): "Say Austin" collapses; top output becomes generic completions like " a" or " home".
- Ablate "Texas" (`-2x`): "Say Austin" no longer wins; other capital tokens (Houston, Dallas itself) appear near the top - the model still knows to output a city but has lost which state.
- Ablate "capital" (`-2x`): "Say a capital" collapses, which in turn collapses "Say Austin" - the two-hop dependency is visible in the graph.
- Ablate "state" (`-2x`): "Texas" weakens; the country-vs-state ambiguity in the prompt now favors the "city name" latents alone.

**Cross-prompt feature swapping**: implemented `cross_prompt_swap(base_prompt, swap_prompt, features_off, features_on)`. Runs `swap_prompt` to get its activations, then in the base prompt zero-ablates `features_off` and injects `scale * swap_activations[features_on]`. Two cases tested:
- Base: `"...containing Dallas is"`, swap: `"...containing Oakland is"`. Turn off Texas features, turn on California features. Model now predicts " Sacramento".
- Base: same Dallas prompt, swap: `"...containing Shanghai is"`. Turn off Texas, turn on China. Model predicts " Beijing".

**Open-ended generation with sustained interventions**: for multi-token generation, wrapped fixed-position interventions with `slice(seq_len-1, None, None)` so the intervention fires at *every* newly-generated token position, not just the initial forward. `model.feature_intervention_generate(prompt, interventions, do_sample=False, max_new_tokens=15)`. Compared pre-text (baseline generation) vs post-text (with Texas ablated across all steps).

```python
from circuit_tracer import ReplacementModel, attribute
from collections import namedtuple

replacement_model = ReplacementModel.from_pretrained(
    "google/gemma-2-2b", "gemma", dtype=t.bfloat16, backend="transformerlens",
)

# Full pipeline in two lines.
dallas_graph = attribute("Fact: the capital of the state containing Dallas is",
                         replacement_model, verbose=True)
logits, activations = replacement_model.get_activations(
    "Fact: the capital of the state containing Dallas is", sparse=True,
)


# Supernode intervention: -2x the default = "strongly suppress this concept".
Intervention = namedtuple("Intervention", ["supernode", "scaling_factor"])

def supernode_intervention(model, graph, interventions):
    intervention_tuples = []
    for inv in interventions:
        for i, feat in enumerate(inv.supernode.features):
            default = inv.supernode.default_activations[i].item()
            intervention_tuples.append((*feat, inv.scaling_factor * default))
    new_logits, new_activations = model.feature_intervention(graph.prompt, intervention_tuples)
    return new_logits, new_activations


# Cross-prompt swap: mix features from two different prompts.
def cross_prompt_swap(model, base_prompt, swap_prompt, features_off, features_on, scale=2.0):
    _, swap_acts = model.get_activations(swap_prompt, sparse=True)
    interventions = [(*f, 0.0) for f in features_off]
    interventions += [(*f, scale * swap_acts[f]) for f in features_on]
    _, modified = model.feature_intervention(base_prompt, interventions)
    return modified


# Sustained interventions during multi-token generation.
def generate_with_intervention(model, prompt, interventions, max_new_tokens=20):
    seq_len = len(model.tokenizer(prompt).input_ids)
    open_interventions = [
        (layer, slice(seq_len - 1, None), feat_idx, value)
        for layer, pos, feat_idx, value in interventions
    ]
    baseline = model.feature_intervention_generate(prompt, [], do_sample=False,
                                                    max_new_tokens=max_new_tokens)[0]
    steered = model.feature_intervention_generate(prompt, open_interventions, do_sample=False,
                                                   max_new_tokens=max_new_tokens)[0]
    return baseline, steered
```

### Results
- All four predicted ablation effects match. The circuit is genuinely causal, not just correlational.
- Cross-prompt swap: replacing Texas with California/China correctly redirects the output to Sacramento/Beijing. This is strong evidence that "Texas" is not just a co-occurring signal - it's a compositional slot that other state features can fill.
- Sustained interventions during generation: with Texas ablated at every step, the model doesn't just fumble the first token - it generates a coherent alternative continuation that avoids Texas-related concepts throughout.

### Learning
- **Predict-then-test** is the discipline that makes ablation experiments informative. Writing down expected effects first catches confirmation bias; when the observed effect diverges from the prediction, that's the interesting data.
- **Compositional slot-filling** (via cross-prompt swap) is a stronger causal test than ablation. Ablation shows a feature is necessary; slot-filling shows *what role* it plays in the computation.
- **Slices for sustained interventions**. Multi-token generation forwards one token at a time; a fixed-position intervention only fires on the initial forward. Wrapping the position in `slice(seq_len-1, None, None)` makes it fire at every "next-token" position throughout generation.
- **The circuit-tracer library removes the graph-building boilerplate** but doesn't change the underlying story. Every line of Section 3's implementation is doing something the library abstracts - which is why the section 3 exercises exist in the first place.

---

## 5. Bonus: manual attribution graphs

An alternative to gradient-based edge computation: build the adjacency matrix by explicitly propagating attribution vectors through the linearized model, component by component. Slower but more transparent - each edge weight comes from a specific matrix product, not from a gradient computation.

### Tasks
- Implemented three propagation helpers on the frozen model:
  - `map_through_ln(x, cache, model, layer)`: apply frozen RMSNorm - since the scale is cached, this is just `x / cached_scale * weight`, a linear operation.
  - `map_through_attn(x, cache, model, layer)`: apply frozen attention as a linear mixing of positions using the cached patterns. `x` (a `(seq, d_model)` attribution vector) goes through `W_V`, gets mixed by the frozen pattern, then `W_O`.
  - `map_through_mlp(x, cache, model, layer, transcoders)`: apply the transcoder's linear skip: `x @ W_skip` (or the transcoder's linearization at the cached activation point).
- Implemented `compute_adjacency_matrix_manual`: for each target node, start with its reading vector, walk backward through the model applying LN/attn/MLP maps at each layer, dot with every source node's writing vector to get the edge weight. Verified this matches the gradient-based result from Section 3 to within float precision.

### Learning
- The gradient-based approach (Section 3) and the manual matrix-product approach are **mathematically equivalent for a linearized model**. The gradient just automates the chain rule that the manual version writes out explicitly.
- Manual attribution is useful for debugging: if `map_through_attn(reading_vec, layer=L)` produces something unexpected, the responsible cached pattern can be traced back. With gradients, the intermediate quantities are hidden.

---

## Takeaways

1. **Latents alone are not a circuit.** A monosemantic feature basis is a starting point; understanding how features compose across layers is what's actually needed for a mechanistic explanation. Every method in this notebook is trying to close that gap.
2. **Three ways to compute cross-layer edges, escalating in generality**: (a) Jacobian of active-latent-to-active-latent on a specific input (Section 1), (b) input-free pullback between transcoder decoder/encoder vectors (Section 2), (c) linearized-model gradient attribution over the whole graph (Section 3). Each generalizes the last.
3. **Transcoders > standard MLP-SAEs for cross-layer analysis.** Both encoder and decoder in residual-stream space -> dot products across layers are well-defined without a forward pass. That's the entire reason attribution graphs use transcoders instead of MLP-out SAEs.
4. **Linearize while preserving the forward pass.** The skip-connection MLP trick, frozen attention patterns, and frozen LayerNorm scales together make the model's residual stream linear on the backward pass while leaving forward logits identical. This is what makes gradient-based attribution *exact* rather than approximate.
5. **Reading/writing vectors are the unifying abstraction.** Every node type (embedding, latent, MLP error, logit) has a `d_model` vector for what it writes into the residual stream and one for what it reads. All edges are dot products in this frame.
6. **Tied embeddings pollute naive de-embedding.** GPT-2's `W_E = W_U.T` means `W_E @ W_enc[latent]` mixes "what activates this latent" with "what this latent's read direction resembles at the output." The extended embedding through layer-0 MLP fixes this.
7. **Predict-then-test on interventions.** The predict-then-test discipline is how real understanding gets separated from post-hoc rationalization. If four ablation predictions all match, the circuit is real; if two out of four match, something is wrong with the story.
8. **Cross-prompt slot-filling is the strongest causal test.** Ablation shows necessity. Slot-filling shows compositional role. If turning off Texas + turning on California cleanly redirects the answer to Sacramento, the "state" slot is genuinely a slot, not a coincidence.

---

## Mini-glossary

- **SAE (Sparse Autoencoder)**: reconstructs one activation site; both encoder and decoder are on that site. For MLP-out SAEs the encoder is in MLP-hidden space.
- **Transcoder**: an SAE-shaped module that replaces an MLP end-to-end. Encoder and decoder both live in residual-stream (`d_model`) space, which is what enables input-free cross-layer analysis.
- **Latent gradient**: the Jacobian `d(upper_latent) / d(lower_latent)` computed via `torch.func.jacrev`, restricted to active latents only via a sparse representation.
- **Pullback**: `sae_lower.W_dec[i] @ sae_upper.W_enc[:, j]` - an input-free measure of how much latent `i` writes into a direction latent `j` reads.
- **De-embedding**: `W_E @ W_enc[:, latent]` - which tokens' embeddings most activate this latent. Compare with **logit lens** (`W_dec[latent] @ W_U`) which asks the reverse question.
- **Extended embedding**: `W_E + MLP_0(W_E)` (approximately). Breaks GPT-2's tied-embedding degeneracy so de-embedding results become semantically meaningful.
- **Attribution graph**: nodes are latents/embeddings/MLP-errors/logits; edges are `source_write @ linearized_flow @ target_read`. Built globally on a linearized model.
- **Local replacement model**: the linearized-forward-pass model used in attribution graphs. Preserves the real forward pass (via the skip-connection trick on MLPs) but makes the backward pass linear.
- **Freeze hooks / TranscoderReplacementHooks**: the two hook classes that implement linearization. FreezeHooks caches attention + LN, TranscoderReplacementHooks routes MLP gradients through the linear surrogate.
- **Reading vector / Writing vector**: for a node, the `d_model` direction it reads to fire (encoder-side) and the direction it writes when it fires (decoder-side * activation).
- **Salient logit**: a top-predicted output token whose `demeaned W_U` column becomes an output-node reading vector.
- **Influence**: total direct + indirect effect of a node on the logits, computed via a truncated Neumann series `I = (I + A + A^2 + ...) @ logit_weights`.
- **Supernode**: a manually-defined group of features that share a conceptual role (e.g. all the "Texas" features). Used for higher-level ablations.
- **`ReplacementModel`**: circuit-tracer's model wrapper that comes pre-loaded with transcoders + freeze hooks. Has `.feature_intervention(prompt, interventions)` and `.feature_intervention_generate(...)` on top of the underlying HookedSAETransformer.
- **Cross-prompt feature swap**: turn off features from prompt A, turn on features from prompt B at their prompt-B activation levels. Tests whether a set of features fills a compositional slot.
- **MLP error node**: the residual `mlp_out - transcoder(mlp_in)` that the transcoder failed to reconstruct. Kept as an explicit node so attribution graphs stay honest about unexplained flow.
