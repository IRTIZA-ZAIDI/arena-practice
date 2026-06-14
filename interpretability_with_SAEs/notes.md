# Interpretability with SAEs - Notes

Notes from ARENA [1.3.3] Interpretability with SAEs. Companion to [1_3_3_Interpretability_with_SAEs_exercises.ipynb](1_3_3_Interpretability_with_SAEs_exercises.ipynb).

The central question: can a **sparse autoencoder** decompose the residual stream into interpretable, monosemantic features that map to circuits we can actually reason about? The exercises go end-to-end: load pretrained SAEs from SAELens, replicate Neuronpedia-style dashboards, run causal-attribution experiments on the IOI task with SAE latents, steer a model with an SAE decoder direction, and finally train + evaluate a small SAE from scratch.

Tool: `sae_lens` (SAEs, `HookedSAETransformer`, `ActivationsStore`), built on top of TransformerLens. The `HookedSAETransformer` wraps a normal `HookedTransformer` but exposes `run_with_cache_with_saes`, `run_with_saes`, and a `with model.saes(...)` context manager so SAEs can be attached / detached without changing the model itself.

Models used: GPT-2-small (most exercises), Gemma-2-2B (GemmaScope feature steering), Gemma-2B-IT (patch scoping), TinyStories-1L (training).

---

## Big picture (what an SAE *is*, operationally)

- An SAE is a **wide, sparse, two-layer net** trained to reconstruct activations from one site in a transformer (e.g. `blocks.7.hook_resid_pre`):
  - `z = ReLU(W_enc @ (x - b_dec) + b_enc)` (latents - shape `d_sae`, typically 16x-65x `d_model`)
  - `x_hat = W_dec @ z + b_dec` (reconstruction)
  - Loss = `||x - x_hat||^2 + lambda * ||z||_1` (reconstruction + L1 sparsity).
- The hypothesis being tested: **the model represents features in a superposed, polysemantic way**, but those features are *sparse* across the dataset. An overcomplete sparse code recovers them as `d_sae` interpretable directions in residual-stream space.
- Two privileged objects per latent:
  - `W_enc[:, i]` - encoder direction. What pattern in `x` makes latent `i` fire?
  - `W_dec[i, :]` - decoder direction. What gets *written* to the residual stream when latent `i` fires?
- Interpretability claim: a *good* latent has a **monosemantic** activation pattern (same concept across all max-activating examples) and a **consistent functional effect** (its decoder direction does one thing when projected through downstream weights).

---

## 1. SAE basics, dashboards, attention SAEs, IOI

Section 1 is the long section. Six sub-themes: loading and running SAEs, building Neuronpedia-style latent dashboards (histogram + max-activating examples + top/bottom logits + autointerp), attention SAEs and DFA, finding name-mover features for IOI via two routes (max activations vs DLA), causal validation via ablation, and faster attribution patching.

### Tasks
- Loaded a pretrained residual-stream SAE for GPT-2-small at layer 7:
  ```python
  gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
  gpt2_sae, gpt2_sae_cfg, gpt2_sae_sparsity = SAE.from_pretrained(
      release="gpt2-small-res-jb", sae_id="blocks.7.hook_resid_pre", device=str(device))
  ```
- Attached the SAE during a forward pass and confirmed reconstruction is near-lossless when `use_error_term=True` (the SAE error is added back) and slightly degrades the model when `use_error_term=False` (only the reconstruction is passed through).
- Built an `ActivationsStore` to stream activations from `NeelNanda/pile-10k` for dashboards:
  ```python
  gpt2_act_store = ActivationsStore.from_sae(
      model=gpt2, sae=gpt2_sae, dataset="NeelNanda/pile-10k",
      streaming=True, store_batch_size_prompts=16, n_batches_in_buffer=32,
      device=str(device),
  )
  ```
- Implemented `show_activation_histogram` (frequency-of-activation + magnitude histogram, restricted to positive acts).
- Implemented `fetch_max_activating_examples`: stream batches, cache only `hook_sae_acts_post`, pick top-k tokens by activation, slice ±buffer context around each. Wrote a non-overlapping variant using `get_k_largest_indices(..., no_overlap=True)` so the top-k don't cluster on the same sequence.
- Implemented `show_top_logits`: the top / bottom tokens by `W_dec[latent] @ W_U` (what does this latent push for / against in the unembed?).
- Implemented an **autointerp** pipeline: send the top-k examples to an LLM (OpenAI / Gemini), ask it for a one-line description, then validate by scoring held-out examples.
- Loaded **attention SAEs** (`gpt2-small-hook-z-kk`) at every layer. These SAEs decompose `hook_z` (per-head output before `W_O`) rather than the residual stream.
- Implemented **Direct Feature Attribution (DFA)** for attention SAEs: the contribution from a source token to a destination latent firing = `attn_weight[dest, src] * v[src] @ W_enc_for_latent`. Surfaces both *which destination position* a latent activates at and *which source position* drives it.
- IOI case study (the running example): prompts like `"When John and Mary went to the shops, John gave the bag to"` -> the correct next token is the indirect object (here `" Mary"`). Two routes to find name-mover features:
  - **Max activations**: at the final token, sort latents by activation. Identifies *active* latents but many will be off-topic.
  - **Direct Logit Attribution (DLA)**: project the latent's decoder direction through `W_O` (since it's an attention-SAE latent) and then through `W_U[correct] - W_U[incorrect]`. Multiply by the latent's actual activation. This finds latents whose decoder direction *specifically* pushes IO > S.
- Implemented `ablate_sae_latent` for **causal validation**: zero a single latent at a specific sequence position, rerun, measure logit-diff drop. Ranks latents by causal effect, not correlation.
- Implemented **attribution patching** as a fast approximation: gradient of the metric w.r.t. latent acts, multiplied by the activations themselves. Avoids the `d_sae`-many separate forward passes that brute-force ablation requires.

```python
# Building the activations store + reading per-latent activations.
def fetch_max_activating_examples(model, sae, act_store, latent_idx,
                                  total_batches=100, k=10, buffer=10):
    """Top-k token positions for `latent_idx`, with ±buffer context."""
    data = []
    sae_acts_post_hook = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    sae.use_error_term = False
    for _ in range(total_batches):
        tokens = act_store.get_batch_tokens()
        _, cache = model.run_with_cache_with_saes(
            tokens, saes=[sae],
            stop_at_layer=sae.cfg.hook_layer + 1,    # save compute past the SAE
            names_filter=[sae_acts_post_hook],
        )
        # cache shape: (batch, seq, d_sae) -> pick one latent: (batch, seq).
        sae_acts = cache[sae_acts_post_hook][:, :, latent_idx]
        idx = get_k_largest_indices(sae_acts, k=k, buffer=buffer)
        toks_with_buf = index_with_buffer(tokens, idx, buffer=buffer)
        for (b, s), context in zip(idx.tolist(), toks_with_buf):
            data.append((sae_acts[b, s].item(), model.to_str_tokens(context), buffer))
    return sorted(data, reverse=True)[:k]


def show_top_logits(model, sae, latent_idx, k=10):
    """Top / bottom unembed-projected tokens for one SAE latent."""
    logits = sae.W_dec[latent_idx] @ model.W_U                # (d_vocab,)
    pos_logits, pos_ids = logits.topk(k)
    neg_logits, neg_ids = logits.topk(k, largest=False)
    # ... tabulate and print.


# IOI: name-mover features via DLA on attention-SAE latents.
# attn_saes[layer] is (sae_obj, sae_cfg, sparsity).
sae_obj, sae_cfg, _ = attn_saes[9]
hook_name = sae_cfg["hook_name"]                              # "blocks.9.attn.hook_z"

_, cache = gpt2.run_with_cache_with_saes(prompts, saes=[sae_obj])
sae_acts_post = cache[f"{hook_name}.hook_sae_acts_post"][:, -1]   # (batch, d_sae)

# Logit direction at the final position: IO > S.
logit_dir = gpt2.W_U.T[correct_toks] - gpt2.W_U.T[incorrect_toks]  # (batch, d_model)

# Push each latent's z-space decoder through W_O to get residual-stream space,
# then dot with logit_dir, then multiply by the actual activation to get DLA.
W_O_layer = gpt2.W_O[9]                                            # (n_heads, d_head, d_model)
# Reshape decoder from (d_sae, n_heads * d_head) -> (d_sae, n_heads, d_head) and mix heads.
dla = ...   # the einsum that finishes the DLA computation


# Brute-force ablation: zero one latent at one position, measure logit-diff drop.
def ablate_sae_latent(sae_acts, hook, latent_idx=None, seq_pos=None):
    sae_acts[:, seq_pos, latent_idx] = 0.0
    return sae_acts

# Only ablate "alive" latents (any positive act at s2_pos across prompts).
acts = cache[hook_sae_acts_post]
alive_latents = (acts[:, s2_pos] > 0).any(dim=0).nonzero().flatten().tolist()
ablation_effects = t.zeros(d_sae, device=acts.device)
for latent in alive_latents:
    fn = partial(ablate_sae_latent, latent_idx=latent, seq_pos=s2_pos)
    abl_logits = gpt2.run_with_hooks_with_saes(
        prompts, saes=[sae_obj], fwd_hooks=[(hook_sae_acts_post, fn)],
    )
    ablation_effects[latent] = clean_logit_diff - logits_to_ave_logit_diff(abl_logits)


# Attribution patching: ablation ≈ activation * gradient. One forward + backward,
# not d_sae forwards. Approximate but much cheaper.
def get_cache_fwd_and_bwd(model, saes, input, metric):
    filter_sae = lambda name: "hook_sae_acts_post" in name
    cache_dict = {"fwd": {}, "bwd": {}}
    def hook(act, hook, dir):
        cache_dict[dir][hook.name] = act.detach()
    with model.saes(saes=saes):
        with model.hooks(
            fwd_hooks=[(filter_sae, partial(hook, dir="fwd"))],
            bwd_hooks=[(filter_sae, partial(hook, dir="bwd"))],
        ):
            metric(model(input)).backward()
    return (ActivationCache(cache_dict["fwd"], model),
            ActivationCache(cache_dict["bwd"], model))
```

### Results
- **Dashboards work**: max-activating examples for individual latents look monosemantic (e.g. latent 9 fires on tokens related to time / chronology; latent 16873 on a different specific concept). The top-logit view explains *what each latent pushes for* once it writes to the residual stream.
- **Name-mover features for IOI (layer 9 attention SAE)**:
  - Max-activations alone surfaces some real name movers but also unrelated active latents.
  - DLA isolates them much more cleanly. The top-2 DLA latents are the "real" name movers; the rest are 10x weaker.
  - Ablation confirms causality: zeroing the top DLA latents at `s2_pos` collapses the IOI logit diff.
- **Attribution patching** correlates strongly with ablation effects but is orders of magnitude faster. Disagreements (off-diagonal points in the scatter) are usually large negative-effect latents where the linear approximation breaks down.
- **Sentiment task replication**: same DLA recipe on a prompt like `'John says, "I want to be alone right now." John feels very'` surfaces latents that push `" sad"` > `" happy"`.

### Learning
- **`stop_at_layer=sae.cfg.hook_layer + 1`** is the magic kwarg for dashboard-building. The model only runs up to the SAE site, so per-batch forward cost is roughly proportional to layer depth, not full model depth.
- **`use_error_term`** controls reconstruction fidelity: `True` adds `(x - x_hat)` back into the residual stream (lossless downstream), `False` uses the reconstruction alone (slightly degrades the model but is what most SAE-interp experiments assume).
- `hook_sae_acts_post` is the **post-ReLU latent activations**, what to read for "did this latent fire?" Everything downstream (max-activating examples, ablation, attribution patching) lives on this hook.
- **Activation > causal effect.** A latent can fire strongly without being causally important. DLA + ablation is what separates "active" from "useful."
- For attention SAEs, the decoder vector lives in **z-space** (`d_model = n_heads * d_head`), not residual space. Push through `W_O` to translate before dotting with anything in residual-stream coordinates.
- **DFA** (`attn_weight * value @ W_enc_latent`) is the attention-SAE analog of attribution. It explains *what source token drove the latent to fire at the destination token*.
- Brute-force ablation is honest but slow (`d_sae` forward passes per layer). Attribution patching is the gradient-based shortcut: one forward + backward gives an approximation to the full ablation map. Use attribution patching to filter; ablate the top candidates.

---

## 2. Understanding latents - feature splitting, autointerp, patch scoping

Section 2 zooms in: not all latents are clean. Some are **polysemantic** (split across two unrelated concepts), some are **redundant** (one concept absorbed across multiple latents), and the more reliable interpretations need a held-out validation step.

### Tasks
- Loaded a stack of "**feature-splitting** SAEs" - the same site in GPT-2-small at multiple widths (768, 1536, 3072, 6144, ...). The hypothesis: wider SAEs *split* a single broad concept into several narrower ones.
- Used three approaches to detect splitting:
  - **Approach 1**: for one latent in the narrow SAE, find wide-SAE latents that fire at the same token positions.
  - **Approach 2**: run a prompt through both SAEs, compare which latents fire on the same tokens.
  - **Approach 3**: search **autointerp descriptions** (downloaded from Neuronpedia) for a keyword (e.g. `"Emotions"`) across both SAEs.
- Built a **UMAP visualization** of decoder vectors across SAE widths, colored by HDBSCAN cluster. Wider SAEs' latents cluster tighter and split narrow-SAE latents into finer-grained groups.
- Implemented an **autointerp scorer**: for each latent, split max-activating examples into "explanation" and "scoring" sets; let an LLM generate a description from the first set; then have the LLM rate held-out examples ("is this an instance of the described pattern?"). A description that scores well generalizes; one that scores poorly is post-hoc rationalization.
- Implemented **patch scoping**: ask an instruction-tuned model `'What is the meaning of "X"?'`, then *replace the residual stream at the `X` position* with the SAE latent's (scaled, normalized) decoder direction. The model's verbose answer describes what the latent represents *in the model's own words*.
- Did **scale tuning** for patch scoping: cosine similarity between the injected direction at `replacement_layer=2` and the residual at `diagnostic_layer=15` peaks at a specific scale (~17-20 for Gemma-2B-IT). Below that the injection is too weak, above it the model loses coherence.

```python
# Autointerp scoring loop.
class AutoInterp:
    def __init__(self, cfg, model, sae, act_store, api_key):
        ...

    def gather_examples(self, latent):
        """Top-k activating + random examples, split into explain/score halves."""
        ...

    def explain(self, examples_for_explain):
        """LLM call: 'Describe what these examples have in common.'"""
        messages = [{"role": "system", "content": SYS_EXPLAIN},
                    {"role": "user",   "content": format(examples_for_explain)}]
        return call_llm(messages)

    def score(self, explanation, examples_for_score):
        """LLM call: 'Rate each example 1-10 for fit.'"""
        ...

    def run(self, debug=False):
        results = {}
        for latent in self.cfg.latents:
            explain_ex, score_ex = self.gather_examples(latent)
            desc = self.explain(explain_ex)
            score = self.score(desc, score_ex)
            results[latent] = (desc, score)
        return results


# Patch scoping (instruction-tuned models): replace residual at "X" position with the latent direction.
def hook_fn_patch_scoping(activations, hook, seq_pos, latent_vector):
    # Only replace on the first forward (full prompt). KV-cached subsequent calls
    # have shape (1, 1, d_model), which we leave alone.
    if activations.shape[1] > 1:
        activations[:, seq_pos] = latent_vector


def generate_patch_scoping_explanation(model, sae, prompt, latent_idx,
                                       replacement_layer, scale, max_new_tokens=50):
    positions = [
        i for i, a in enumerate(model.tokenizer.encode(prompt))
        if model.tokenizer.decode([a]) == model.tokenizer.unk_token
    ]
    latent_dir = sae.W_dec[latent_idx]
    latent_dir_scaled = (latent_dir / latent_dir.norm(dim=-1)) * scale
    hook = partial(hook_fn_patch_scoping,
                   latent_vector=latent_dir_scaled, seq_pos=positions)

    with model.hooks(fwd_hooks=[(get_act_name("resid_pre", replacement_layer), hook)]):
        return model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)
```

### Results
- **Feature splitting is real**. A narrow-SAE latent labeled "Emotions" maps to ~5-10 wide-SAE latents labeled "sadness", "anger-language", "anxiety-vocabulary", etc. UMAP shows nested cluster structure at increasing SAE width.
- **Autointerp scoring catches bad descriptions**. Some latents that *look* monosemantic in the top-10 activations don't generalize; their score on held-out examples is near random. A latent's interpretability is the score, not the impression.
- **Patch scoping produces fluent, accurate self-explanations** when the scale is tuned correctly. For a latent that fires on `" wedding"`-related context, the model produces something like `"The word 'X' refers to a ceremony of marriage and the events surrounding it..."`.
- Scale-tuning curve has a clear sweet spot: too low = the injection is ignored; too high = generations become repetitive or off-topic.

### Learning
- **Width vs interpretability is non-monotonic.** Going wider splits features cleanly up to a point, then introduces **dead latents** (never fire) and **redundant latents** (multiple firing on the same pattern). The optimum is dataset-dependent.
- **Held-out scoring is the only honest interp metric.** Eyeballing the top-10 activations is fast but biased; the LLM-as-scorer at least makes the bias explicit.
- **Patch scoping is the cheapest "decoded vocabulary" probe.** Instead of dotting decoder against `W_U` and reading top tokens (which can be misleading because the decoder doesn't write directly to logits), inject the direction and let the model verbalize.
- **KV-caching breaks naive interventions.** Patch-scoping's hook has to check `activations.shape[1] > 1` and only patch on the first forward pass; subsequent generation steps have cached past keys/values and only forward the new token.

---

## 3. Training & evaluating SAEs

The third section drops the pretrained-SAE assumption and trains one from scratch on a tiny model (TinyStories-1L-21M), MLP-out site. The goal is to see what the training curves look like and what counts as "trained well."

### Tasks
- Set up `SAETrainingRunner` with `LanguageModelSAERunnerConfig`. Key knobs:
  - `d_in`: size of the residual / MLP-out site (matches the model).
  - `expansion_factor`: `d_sae / d_in` (typical: 16x-65x).
  - `l1_coefficient`: weight on the L1 sparsity term. Warm-up over the first 10% of training prevents collapse.
  - `lr` + warm-up + decay: standard AdamW schedule.
  - `training_tokens`: ~`30_000 * 4096 ≈ 1.2e8` for the demo run.
- Trained a TinyStories-1L SAE; visualized features with `sae_vis.SaeVisData`.
- Worked through the **diagnostic plots** for good vs bad training runs:
  - Reconstruction loss should drop quickly then plateau.
  - L0 (number of active latents per token) should drop to a small fraction of `d_sae` (e.g. 10-100 out of thousands).
  - Dead-latent fraction should stay low; large dead fractions indicate l1 too high.
  - Cross-entropy-loss-with-vs-without-SAE should converge close to the base model's loss.

```python
from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig

total_training_steps = 30_000
batch_size = 4096
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = l1_warm_up_steps = total_training_steps // 10        # 10% warm-up
lr_decay_steps   = total_training_steps - lr_warm_up_steps - 1000        # decay rest

cfg = LanguageModelSAERunnerConfig(
    model_name="tiny-stories-1L-21M",
    hook_name="blocks.0.hook_mlp_out",
    d_in=1024,
    expansion_factor=16,
    l1_coefficient=5.0,
    lr=5e-5,
    lr_warm_up_steps=lr_warm_up_steps,
    lr_decay_steps=lr_decay_steps,
    l1_warm_up_steps=l1_warm_up_steps,
    training_tokens=total_training_tokens,
    train_batch_size_tokens=batch_size,
    ...
)
runner = SAETrainingRunner(cfg)
trained_sae = runner.run()


# Inspect the trained SAE with sae_vis.
from sae_lens import SAE
from sae_vis import SaeVisData, SaeVisConfig

tokens = t.tensor([x["input_ids"] for x in dataset["train"].take(1024)], device=device)
sae_vis_data = SaeVisData.create(
    sae=trained_sae,
    model=tinystories_model,
    tokens=tokens,
    cfg=SaeVisConfig(features=range(16)),
    verbose=True,
)
sae_vis_data.save_feature_centric_vis(filename="feature_vis.html", verbose=True)
```

### Results
- TinyStories-1L SAE produces obvious clean features after ~30k steps: tokens like proper names cluster on a few specific latents, repeated patterns from the dataset (`"Once upon a time"`) get their own latents.
- The W&B-style curves match the diagnostic shape: L0 starts high (~`d_sae` early in training) and decays to ~50-200 active latents per token; reconstruction MSE drops fast, then plateaus; CE-with-SAE tracks within 1-2% of CE-without-SAE.

### Learning
- **L1 warm-up matters.** Starting l1 at zero and ramping up over the first 10% of training prevents the encoder from collapsing all latents to zero in the first few steps (which it does if l1 is full-strength from step 0, because the gradient signal from reconstruction is too weak early).
- **Two losses, one tradeoff.** Reconstruction loss alone -> dense, uninterpretable latents. Pure L1 -> all zeros, perfect sparsity, no information. The whole interesting part of the design space is in tuning `l1_coefficient` to land in the regime where both are non-trivial.
- **Evaluation isn't just reconstruction MSE.** A useful SAE needs: (1) low reconstruction error, (2) low L0 (sparsity), (3) low fraction of dead latents, (4) downstream CE-loss-with-SAE close to clean. Optimizing only (1) gives a useless dense autoencoder.

---

## Takeaways

1. **SAEs are decompositions, not classifiers.** The output is a sparse vector of latent activations interpreted as "which underlying features fired here." All downstream analysis (dashboards, DLA, ablation) treats latents as if they were neurons, but they're learned axes of a sparse basis, not units the model itself uses.
2. **`hook_sae_acts_post` is the central object.** Max-activating examples, ablation, attribution patching, autointerp - all of them read this single hook. Knowing how to cache it cheaply (with `stop_at_layer` and `names_filter`) is the whole game.
3. **Activation alone is not interpretation.** A latent firing strongly on a context doesn't mean it represents that context causally. Always check via DLA, ablation, or patch scoping before claiming a latent "is" feature X.
4. **DLA + ablation > inspection.** The IOI section is the template: rank latents by Direct Logit Attribution (correlational but cheap and signed), then confirm the top candidates via ablation (causal but slow). Attribution patching is the gradient-based middle ground that scales.
5. **Attention-SAE latents live in z-space.** Their decoder vectors need `W_O` applied before they can be compared with residual-stream things. Forgetting this is a silent dimensional mistake that gives wrong-but-plausible numbers.
6. **Feature splitting is a fact of life.** Wider SAEs decompose broad concepts into finer-grained ones. Pick a width based on the goal: for circuits, narrower is usually fine; for fine-grained semantic distinctions, go wider.
7. **Held-out scoring is the only autointerp metric that counts.** The LLM-as-scorer catches descriptions that fit the top-10 examples but don't generalize. Treat the score, not the description, as the evidence.
8. **Patch scoping is the cleanest "what does this latent mean" probe.** It uses the model's own language faculty to verbalize the direction, instead of indirectly via top unembed tokens.
9. **Training an SAE is mostly about tuning l1 and warm-up.** Get those right and reconstruction follows. The bad-training failure modes (dead latents, dense activations, collapsed reconstruction) all trace back to mis-tuned sparsity.

---

## Mini-glossary

- **SAE (Sparse Autoencoder)**: a 2-layer net trained to reconstruct activations from a transformer hook site, with an L1 penalty on the latent layer. Overcomplete (`d_sae >> d_in`).
- **Latent**: one dimension of the SAE's hidden layer. Sometimes called a "feature" but feature implies interpretability that isn't guaranteed.
- **`HookedSAETransformer`**: SAELens wrapper around `HookedTransformer` that adds `run_with_cache_with_saes`, `run_with_saes`, `with model.saes(...)`.
- **`ActivationsStore`**: streams activations from a dataset for training or for dashboard-style batched inspection.
- **`hook_sae_acts_post`**: the post-ReLU latent activations. The standard hook to cache.
- **`use_error_term`**: if `True`, adds `(x - x_hat)` back into the residual stream so the SAE is invisible to downstream layers. If `False`, the model sees only the reconstruction.
- **L0**: the number of nonzero latent activations per token. Sparsity metric.
- **Dead latent**: one that never fires across a training-distribution sample. Usually means the encoder direction lost its gradient signal during training.
- **DLA (Direct Logit Attribution)**: project a latent's decoder vector through the unembed (and `W_O` for attention SAEs), dot with a logit-difference direction, scale by the activation. Per-latent contribution to a logit difference.
- **DFA (Direct Feature Attribution)**: for attention-SAE latents, `attn_weight[dest, src] * value[src] @ W_enc_latent`. Per source token contribution to a destination latent firing.
- **Attribution patching**: ablation effect ≈ activation * gradient. One forward + backward replaces `d_sae` ablation forwards. Approximate.
- **Patch scoping**: inject the latent's decoder direction at a specific token position of an instruction-tuned model and let the model verbally describe it.
- **Feature splitting**: a single broad concept in a narrow SAE corresponds to multiple finer-grained latents in a wider SAE.
- **Feature absorption**: a single concept is spread across multiple latents in a wide SAE; recovering the original semantics needs OR-ing them together.
- **Autointerp**: LLM-generated description of a latent based on its top activating examples, optionally validated by an LLM scorer on held-out examples.
- **GemmaScope**: open-weights SAEs trained on Gemma-2-2B at many sites and widths.
- **SAELens**: the library (`sae_lens`) that wraps SAE loading, training, and the hooked-model integration.
