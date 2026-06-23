# Indirect Object Identification - Notes

Notes from ARENA [1.4.1] Indirect Object Identification. Companion to [1_4_1_Indirect_Object_Identification_exercises.ipynb](1_4_1_Indirect_Object_Identification_exercises.ipynb).

The central question: when GPT-2-small completes `"When John and Mary went to the shops, John gave the bag to"` with `" Mary"`, **which heads, at which positions, on which paths through the model are responsible**? The IOI paper (Wang et al., 2022) reverse-engineers an end-to-end circuit and the exercises replicate it from scratch: logit attribution -> activation patching -> path patching -> validation of head roles -> minimal-circuit construction.

Model: `gpt2-small` via `HookedTransformer.from_pretrained("gpt2-small", center_unembed=True, center_writing_weights=True, fold_ln=True, refactor_factored_attn_matrices=True)`. All four flags numerically equivalent to vanilla GPT-2 but make interp cleaner.

For a TransformerLens-API cheat sheet referenced throughout these notes, see [transformerlens.md](../transformerlens.md).

---

## Big picture (the IOI circuit)

The paper identifies a circuit of ~26 heads across 7 classes, working in 3 stages:

1. **Duplicate Token Heads** (early, e.g. `0.1`, `3.0`): at the second occurrence of the subject (`S2`), notice that this token already appeared earlier in the sentence. Write "this is a duplicate" into S2's residual stream.
2. **Induction Heads** (early-mid, e.g. `5.5`, `6.9`): similar role - notice repetition via the prev-token-head + induction mechanism from the earlier mech-interp exercises. Together with duplicate-token heads, they signal "S is duplicated."
3. **S-Inhibition Heads** (mid, e.g. `7.3`, `7.9`, `8.6`, `8.10`): read from S2 via Q/K composition with duplicate/induction heads, write *into the END token's residual stream* a signal that says "suppress attention to S, attend to the *other* name (IO) instead." This is the K-composition step that makes the circuit work.
4. **Name Mover Heads** (late, e.g. `9.6`, `9.9`, `10.0`): at END, attend strongly to IO (because S-Inhibition Heads suppressed S), copy IO's token via OV-circuit. Write into the residual in the direction of the IO token's unembed.
5. **Negative Name Movers** (e.g. `10.7`, `11.10`): mirror Name Movers but copy with a *negative* sign. Function unclear but observable.
6. **Backup Name Movers** (10.x): activate if the primary Name Movers are ablated. The model has redundancy built in.
7. **Previous Token Heads** (early, e.g. `2.2`, `4.11`): feed into induction heads.

The interesting findings:
- **Localization is extreme.** Almost all task-relevant computation happens at the **S2** and **END** positions; other tokens barely matter.
- **MLPs barely contribute** (except MLP0, which is essentially an extended embedding).
- The circuit has **redundancy**: ablating a primary head causes backup heads to step in.

---

## 1. Setup, metric, and model flags

### Tasks
- Loaded the model with the four interp-friendly flags. Confirmed `W_Q` and `W_K` columns are orthogonal and equal-norm after `refactor_factored_attn_matrices` (each column is an independent unit-length feature direction).
- Built **8 paired prompts** in ABBA/BABA structure - 4 prompt templates x 2 orderings of names. Each prompt has an associated `answer_tokens` tuple `(IO_token, S_token)` (e.g. `(" Mary", " John")`).
- Defined `logits_to_ave_logit_diff(logits, answer_tokens)`: per-prompt, take `logits[:, -1, IO] - logits[:, -1, S]`. The metric of the whole exercise.

```python
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)

# 8 prompts: 4 templates x 2 name orderings (ABBA + BABA).
prompts = [tmpl.format(name) for tmpl in prompt_format for name in name_pairs[i]]
answer_tokens = ...   # shape (8, 2): (IO_token_id, S_token_id) per prompt

tokens = model.to_tokens(prompts, prepend_bos=True).to(device)
original_logits, cache = model.run_with_cache(tokens)


def logits_to_ave_logit_diff(logits, answer_tokens=answer_tokens, per_prompt=False):
    """Average logit(IO) - logit(S) at the final position."""
    final = logits[:, -1, :]                                            # (batch, d_vocab)
    correct = final.gather(1, answer_tokens[:, 0:1]).squeeze(1)         # logit(IO)
    incorrect = final.gather(1, answer_tokens[:, 1:2]).squeeze(1)       # logit(S)
    diff = correct - incorrect
    return diff if per_prompt else diff.mean()
```

### Learning
- **Logit diff = log-prob diff.** Because `log p(i) - log p(j) = (x_i - x_j) - logsumexp(x)` and the logsumexp cancels, comparing the logits of two specific tokens is exactly the same as comparing their log-probs. This is the foundation of the entire IOI methodology.
- **`fold_ln=True` is safe here** because the IOI exercises lean on linear-decomposition arguments. With folded LN, the only nonlinearity at the unembed boundary is `ln_final.hook_scale` (a single scalar per token position), which is constant when the model is run on a fixed input - so component contributions to the logit diff *are* additive.
- **`refactor_factored_attn_matrices`** rotates `W_Q`/`W_K` so they share the same singular-value basis. Makes per-head analysis cleaner without changing model behavior.

---

## 2. Logit attribution (logit lens, layer, head)

### Tasks
- Computed the **logit diff direction**: `W_U[:, IO] - W_U[:, S]` per prompt. Any residual-stream component's contribution to the IOI metric is its dot product with this direction (after the final LayerNorm scale).
- Used `model.tokens_to_residual_directions(answer_tokens)` to get `(batch, 2, d_model)` then unbinded to get `logit_diff_directions`.
- Implemented `residual_stack_to_logit_diff(residual_stack, cache, logit_diff_directions)`: apply `cache.apply_ln_to_stack(stack, layer=-1, pos_slice=-1)` (the final LayerNorm scale at the final token position), then einsum-dot with the per-batch logit diff direction.
- Built three views via the cache helpers:
  - **Logit lens by layer**: `cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1)` -> cumulative residual after each sublayer.
  - **Per-component**: `cache.decompose_resid(layer=-1, pos_slice=-1)` -> embed + each attn layer + each MLP layer as separate vectors.
  - **Per-head**: `cache.stack_head_results(layer=-1, pos_slice=-1)` -> one vector per `(layer, head)` pair, rearranged to `(layer, head)` shape.

```python
# Logit diff direction lives in residual space.
answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)  # (batch, 2, d_model)
correct_rd, incorrect_rd = answer_residual_directions.unbind(dim=1)
logit_diff_directions = correct_rd - incorrect_rd                                # (batch, d_model)


def residual_stack_to_logit_diff(residual_stack, cache, logit_diff_directions=logit_diff_directions):
    """Project a (..., batch, d_model) stack onto the logit diff direction, after LN scale."""
    batch_size = residual_stack.size(-2)
    # apply_ln_to_stack uses the final layer's hook_scale at the chosen position.
    scaled = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    return einops.einsum(
        scaled, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size


# Logit lens: cumulative contribution as the residual stream accumulates.
acc_resid, labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
logit_lens_diffs = residual_stack_to_logit_diff(acc_resid, cache)

# Per-component (embed, attn_0, mlp_0, attn_1, mlp_1, ...).
per_layer_resid, layer_labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
per_layer_diffs = residual_stack_to_logit_diff(per_layer_resid, cache)

# Per-head: rearrange (n_layers * n_heads, batch, d_model) -> (n_layers, n_heads, batch, d_model).
per_head_resid, head_labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
per_head_resid = einops.rearrange(per_head_resid, "(layer head) ... -> layer head ...",
                                  layer=model.cfg.n_layers)
per_head_diffs = residual_stack_to_logit_diff(per_head_resid, cache)   # (n_layers, n_heads)
```

### Results
- **Logit lens by layer**: logit diff is roughly zero until layer 7, ramps up sharply through layers 8-9, plateaus or slightly drops in layers 10-11.
- **Per-layer**: layers 9 and 10 contribute the most positive logit diff. Layer 11 is slightly negative.
- **Per-head heatmap (the headline plot)**: heads `9.6`, `9.9`, `10.0` dominate positive contributions (Name Movers). Heads `10.7`, `11.10` are strongly negative (Negative Name Movers).

### Learning
- **`apply_ln_to_stack` is the key correctness step.** Without it, decomposing the residual stream into pieces and projecting each onto `logit_diff_direction` is wrong - the final LN scale matters and it's the *same scale* applied to every component (the scale comes from the *full* residual, not from each piece individually).
- **Logit lens is observational, not causal.** It tells what a component is *adding* to the logit diff direction; it says nothing about what would happen if that component were removed.
- **Negative Name Movers are real and consistent.** A small set of late heads systematically push *against* the correct answer. The IOI paper has no clean story for why; it's an observed phenomenon to look out for in any circuit.

---

## 3. Activation patching

The first causal step. Run the model twice: once on `clean_tokens` (real IOI prompts) and once on `corrupted_tokens` (same prompts with names swapped). Then run a third time on the corrupted input, but **patch in one specific activation from the clean cache** and see how much performance recovers.

### Tasks
- Built `corrupted_tokens` by swapping adjacent pairs in `clean_tokens`. Same name set, but the IO/S roles are flipped.
- Computed `clean_logit_diff` (~3.5) and `corrupted_logit_diff` (~-3.5).
- Defined `ioi_metric(logits, ...)`: linear interpolation calibrated so corrupted -> 0, clean -> 1.
  ```python
  def ioi_metric(logits, answer_tokens, corrupted_logit_diff, clean_logit_diff):
      diff = logits_to_ave_logit_diff(logits, answer_tokens)
      return (diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
  ```
- Ran three patching sweeps via `transformer_lens.patching`:
  - `get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, ioi_metric)`: patch each (layer, position) in `resid_pre` -> heatmap (layer x position).
  - `get_act_patch_block_every(...)`: faceted by sublayer (resid / attn_out / mlp_out).
  - `get_act_patch_attn_head_out_all_pos(...)`: patch each head's full output -> (layer x head) heatmap.
- Hand-implemented `patch_residual_component`, `patch_head_vector`, `get_act_patch_block_every` from scratch using forward hooks - same outputs as the library functions, confirms understanding.

```python
def patch_residual_component(corrupted_residual, hook, pos, clean_cache):
    """Replace one (layer, position) slot with the clean value, leave the rest of the corrupted run."""
    corrupted_residual[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_residual


def patch_head_vector(corrupted_head_vector, hook, head_index, clean_cache):
    """Replace one head's output at all positions."""
    corrupted_head_vector[:, :, head_index, :] = clean_cache[hook.name][:, :, head_index, :]
    return corrupted_head_vector


# Library-equivalent patching sweep.
act_patch_resid_pre = patching.get_act_patch_resid_pre(
    model, corrupted_tokens, clean_cache, ioi_metric,
)   # (n_layers, n_positions) heatmap


# Hand-rolled equivalent.
results = t.zeros(model.cfg.n_layers, tokens.shape[1], device=device)
for layer in range(model.cfg.n_layers):
    for pos in range(tokens.shape[1]):
        hook_fn = partial(patch_residual_component, pos=pos, clean_cache=clean_cache)
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(utils.get_act_name("resid_pre", layer), hook_fn)],
        )
        results[layer, pos] = ioi_metric(patched_logits, ...)
```

### Results
- **`resid_pre` heatmap**: a vivid two-stage picture. At the **S2 position**, patching `resid_pre` at layers 0-5 already recovers most of the score (the duplicate-token / induction signal forms early). At the **END position**, patching only recovers performance from layer 7+ (that's when S-Inhibition Heads write the "attend-to-IO" signal into END).
- **Faceted by sublayer**: attention is doing essentially all of the work. MLPs are negligible except MLP0 (which behaves like an extended embedding - the IOI paper calls this out).
- **Per-head heatmap**: the bright pixels are exactly the heads identified by per-head logit attribution: Name Movers in layers 9-10, S-Inhibition Heads in layers 7-8, Negative Name Movers in 10-11.

### Learning
- **The corrupted prompt is the calibration anchor.** Patching scores are scaled by `(clean - corrupted)`. Without a meaningful difference between the two runs, the metric is uninformative.
- **`patching.get_act_patch_*` saves enormous amounts of code** for standard sweeps, but understanding the underlying loop (hook one slot, run forward, measure metric) is what makes custom patching experiments possible.
- **Patching answers "is this activation causal?"** with maximum specificity. A high patching score at `(layer 7, position S2)` says: the computation done by layer 7 at S2 is necessary and sufficient for IOI performance, *conditional on every downstream activation being free to recompute*.
- **`MLP0` is a duplicate of the embedding.** The IOI paper finds that GPT-2-small uses MLP0 as a "second embedding" - presumably because the literal `W_E` lookup is shared between read (input) and write (output) operations and the model wants different geometry for the read.

---

## 4. Path patching

The leap from "is this activation causal?" to "is this specific connection causal?" Activation patching changes one node and lets the effect propagate through everything downstream. Path patching changes one specific edge while *freezing* every other path.

### Tasks
- Loaded the `IOIDataset` class (large bag of templated IOI prompts and an ABC-corrupted variant where both names are replaced).
- Computed `ioi_metric_2(logits)`: 0 on clean IOI, -1 on ABC. Sign-flipped from the section-3 metric.
- Implemented **path patching from each head -> final residual stream** (the Name Mover identification recipe):
  1. Cache activations on `clean_tokens` (orig).
  2. Cache activations on `corrupted_tokens` (new).
  3. For each candidate sender head `(L, H)`: install a `z` hook that patches *that head only* with the corrupted-run value, and *freezes every other head's z* to its orig value. MLPs are NOT frozen (they recompute). Cache the resulting `resid_post[-1]`.
  4. In a fresh clean run, swap in this `resid_post[-1]` and measure metric change.
- Implemented **path patching head -> S-Inhibition heads' value input**: same idea but the "destination" is now the V input of a downstream head, not the final residual.

```python
def patch_or_freeze_head_vectors(orig_head_vector, hook, new_cache, orig_cache, head_to_patch):
    """Patch one head with new_cache, freeze all others to orig_cache."""
    # Freeze all heads to orig (clean) values.
    orig_head_vector[...] = orig_cache[hook.name][...]
    # Then patch the one target head with the new (corrupted) value.
    layer, head = head_to_patch
    if hook.layer() == layer:
        orig_head_vector[:, :, head, :] = new_cache[hook.name][:, :, head, :]
    return orig_head_vector


# Two-pass path patching: each candidate sender -> final residual.
path_patch_results = t.zeros(n_layers, n_heads, device=device)
for sender_layer in range(n_layers):
    for sender_head in range(n_heads):
        # Step 1: run with all-heads-frozen + this one head patched, capture resid_post[-1].
        hook_fn = partial(
            patch_or_freeze_head_vectors,
            new_cache=corrupted_cache, orig_cache=clean_cache,
            head_to_patch=(sender_layer, sender_head),
        )
        fwd_hooks = [(utils.get_act_name("z", L), hook_fn) for L in range(n_layers)]
        _, patched_cache = model.run_with_cache(
            clean_tokens, fwd_hooks=fwd_hooks, names_filter=lambda n: "resid_post" in n)
        patched_resid_final = patched_cache["resid_post", -1]

        # Step 2: in a fresh clean run, swap in just this resid_post[-1].
        def insert_resid(resid, hook, value=patched_resid_final):
            return value
        patched_logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(utils.get_act_name("resid_post", -1), insert_resid)],
        )
        path_patch_results[sender_layer, sender_head] = ioi_metric_2(patched_logits)
```

### Results
- **Head -> final residual path patching**: produces a heatmap almost identical to direct logit attribution from Section 2. This is the right answer: **9.6, 9.9, 10.0** are the Name Mover Heads, **10.7, 11.10** are Negative Name Movers. Path patching is more rigorous than per-head logit attribution because it controls for the head's output being modified by downstream attention.
- **Head -> S-Inhibition Heads' V**: the heads that most strongly drive S-Inhibition V are exactly the duplicate-token heads (`0.1`, `3.0`) and induction heads (`5.5`, `6.9`). This validates the K-composition story: duplicate/induction heads write "S is duplicated" into the residual stream; S-Inhibition Heads read this via their V projections at later layers.

### Learning
- **Activation patching ≠ path patching.** Activation patching of a sender head at a position changes the sender *and lets the effect flow everywhere downstream*. Path patching pins down a specific edge (e.g. sender's output -> final residual) by freezing the alternative paths. They agree when the head's output is mostly used directly (the Name Movers); they disagree when a head's output is consumed by downstream attention before reaching the metric.
- **Why MLPs aren't frozen in path patching**: the IOI paper argues that MLPs are essentially fixed transforms of the residual stream and their effect on the metric is mostly determined by the residual content. Freezing them would over-constrain the experiment.
- **Path patching is a discovery tool, not just a confirmation tool.** Running it head-by-head against a downstream component's V/Q/K input identifies who composes with whom - this is how the duplicate-token and induction heads were linked to the S-Inhibition Heads.

---

## 5. Full circuit replication

### Tasks
- **Writing direction**: for each candidate Name Mover head, scatter `attn[END->IO]` against `(z @ W_O) . W_U[IO]` for each prompt. Repeat for S. A Name Mover should have high attention to IO -> high IO logit contribution, and low attention to S -> low/negative S logit contribution. Replicated paper Figure 3(c).
- **Copying scores**: take `W_E @ W_OV @ W_U` for each `(L, H)`. For a copy head, this matrix should be roughly diagonal (each input token pushes for itself in the output). Score = fraction of vocab where the input token is in the top-k of its own output row. Reproduces paper Figure 3(b) for both positive and negative copying.
- **Head validation**:
  - Duplicate-token heads: high attention from `S2 -> S1`.
  - Induction heads: induction-stripe attention (diagonal offset, like in 1.2).
  - S-Inhibition heads: high `END -> S2` attention.
  - Previous-token heads: diagonal offset by 1.
- **Minimal circuit construction**: ablate all heads NOT in the proposed circuit (mean-ablate, using means computed over the ABC dataset), measure that the truncated model still scores within tolerance of the full one.
- **Minimality scores**: for each head in the circuit, ablate it on top of the truncated model and measure the drop. A high minimality score means the head is genuinely necessary; a low score means redundancy or backup heads compensate.

### Results
- Name Movers `9.6, 9.9, 10.0` and Negative Name Movers `10.7, 11.10` all show the expected scatter patterns - high attention to IO/S correlates with positive/negative logit contribution.
- Copying scores cleanly identify the same set of heads as the writing-direction analysis.
- The 26-head minimal circuit reproduces ~87% of the full model's logit diff. Most individual heads have nonzero minimality scores, but a few (the backup name movers) have small scores because their job is taken over by primaries.

### Learning
- **Multiple lines of evidence are the right standard.** A head is "a Name Mover" only if (1) it has high logit attribution, (2) it survives path patching to the final residual, (3) it attends strongly to IO, and (4) its OV circuit is a near-identity on names. Any one of these alone can be misleading; the conjunction is what counts.
- **Faithfulness, completeness, minimality** are three distinct circuit properties:
  - **Faithfulness**: the circuit alone reproduces the task. (Mean-ablate everything else, check metric recovery.)
  - **Completeness**: no head outside the circuit is necessary. (Add other heads back, check no improvement.)
  - **Minimality**: every head in the circuit is necessary. (Ablate each, check drop.)
- **Backup heads dampen minimality.** The model has redundancy - removing 9.6 causes 10.x backup name movers to step in. This means raw "ablation effect" undercounts the true importance of a node; it's *necessary* only relative to whether backups exist.

---

## 6. Bonus: anomalies and follow-ups

The bonus section pulls on three loose threads from the main circuit:

- Why are some "duplicate-token signal" heads actually **induction heads** (not direct duplicate-token heads)?
- Why does ablating Name Mover `9.9` hurt performance *less than the naive estimate*? (The **backup name movers** result.)
- Are S-Inhibition heads using **token information** about S, or **positional information** about where it sits, or both?

### Tasks

**Early induction heads (`5.5`, `6.9`)**
- Generated repeated-token sequences via `generate_repeated_tokens(model, seq_len, batch)` and visualized attention patterns of `5.5` and `6.9` with `cv.attention.attention_patterns(...)`. Confirmed the induction stripe (attention to the token *after* the previous occurrence) is the same shape as in the [1.2 mech interp](../intro_to_mech_interp/notes.md) exercises - these aren't ad-hoc duplicate-token heads, they're full induction heads doing a more general job.
- Used **path patching from each head to the keys of `5.5` and `6.9`** to identify the previous-token heads that compose with them: confirmed `2.2` and `4.11` are the prev-token heads, exactly as the paper claims.
- Same path-patching recipe pointed at the keys of S-Inhibition heads: surfaces the bracketed `5.8` and `5.9` (the paper's "secondary" induction heads).

```python
import circuitsvis as cv

# Visualize the induction stripe on a repeated-token sequence.
model.reset_hooks(including_permanent=True)
seq_len = 15
rep_tokens = generate_repeated_tokens(model, seq_len, batch_size=1)
_, cache = model.run_with_cache(
    rep_tokens, return_type=None,
    names_filter=lambda name: name.endswith("pattern"),
)
for layer, head in [(5, 5), (6, 9)]:
    display(cv.attention.attention_patterns(
        tokens=model.to_str_tokens(rep_tokens[0]),
        attention=cache["pattern", layer][:, head].unsqueeze(0),
    ))

# Path patching head -> keys of induction heads to find prev-token heads.
induction_key_results = get_path_patch_head_to_heads(
    receiver_heads=[(5, 5), (6, 9)],
    receiver_input="k",                    # patch the key input, not value
    model=model, patching_metric=ioi_metric_2,
)
# Heatmap: heads 2.2 and 4.11 light up.
```

**Backup name movers**
- Ablated head `9.9` at the END position only (the highest-DLA Name Mover). Computed the **naive prediction**: if `9.9` contributes a logit diff of `X` directly, ablating it should drop the total by `X`. Measured: the actual drop is much smaller.
- Re-decomposed per-head contributions to the logit diff post-ablation. The diff `per_head_ablated - per_head_clean` lights up at **layer 10** - the backup name movers (and the previously-negative Negative Name Movers) ramp up to compensate. One Negative Name Mover even reverses sign (becomes less negative).
- Inspected `ln_final.hook_scale` before and after. The scaling factors don't change much, so the rescaling story isn't the explanation - downstream heads really do change their behavior when `9.9` is missing.

```python
# Direct contribution of 9.9 to the logit diff at END.
per_head_residual, labels = ioi_cache.stack_head_results(layer=-1, return_labels=True)
per_head_residual = einops.rearrange(
    per_head_residual[:, t.arange(len(ioi_dataset)), ioi_dataset.word_idx["end"]],
    "(L H) B D -> L H B D", L=model.cfg.n_layers,
)
per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, ioi_cache)

# Now ablate 9.9 at END and rerun.
def ablate_head_at_end(z, hook, head_idx, end_positions):
    z[t.arange(z.shape[0]), end_positions, head_idx] = 0.0
    return z

hook_fn = partial(ablate_head_at_end, head_idx=9,
                  end_positions=ioi_dataset.word_idx["end"])
_, ablated_cache = model.run_with_cache(
    ioi_dataset.toks,
    fwd_hooks=[(utils.get_act_name("z", 9), hook_fn)],
)

# Compare per-head contributions before / after.
per_head_ablated_residual, _ = ablated_cache.stack_head_results(layer=-1, return_labels=True)
# ... same rearrange + residual_stack_to_logit_diff ...
delta = per_head_ablated_logit_diffs - per_head_logit_diffs
# Plot (clean, ablated, delta) side-by-side: layer 10 lights up in delta.
```

**Positional vs token information in S-Inhibition heads**
- Used `IOIDataset.gen_flipped_prompts(spec)` to generate **six dataset variants** crossing token-signal `{same, random, inverted}` with positional-signal `{same, inverted}`:

  | Token signal | Positional signal | `gen_flipped_prompts` spec |
  |---|---|---|
  | same | same | (original) |
  | random | same | `"ABB->CDD, BAB->DCD"` |
  | inverted | same | `"ABB->BAA, BAB->ABA"` |
  | same | inverted | `"ABB->BAB, BAB->ABB"` |
  | random | inverted | `"ABB->CDC, BAB->DCD"` (or similar) |
  | inverted | inverted | `"ABB->BAA, BAB->ABA"` then position swap |

- For each variant: run the model, patch **all S-Inhibition heads' z outputs** simultaneously from the variant's cache into the original IOI run. Measure how the logit diff changes.
- Decomposed per-head: ran the same experiment but patching one S-Inhibition head at a time.

### Results

- **Induction heads in early layers are real induction heads, not duplicate-token heads in disguise.** `5.5`/`6.9` show clean induction-stripe attention on random repeated sequences, and `2.2`/`4.11` light up as their key-side composers in path patching - exactly the two-head induction circuit from the mech-interp exercises.
- **Backup name movers**: ablating `9.9` reduces the logit diff by ~30% of the naive prediction. The "missing" effect is taken up by layer-10 heads. Distinct phenomenon from LN-rescaling (the scaling factors barely change).
- **Positional vs token signal heatmap**:
  - Token-inverted + position-inverted -> logit diff ≈ -clean (full reversal, sanity check).
  - Position-inverted alone destroys performance more than token-inverted alone -> the S-Inhibition heads rely **more on position than on token identity**.
  - Per-head decomposition: head **`8.6`** is *purely* positional (token-inverted dataset doesn't change its effect at all). Other S-Inhibition heads mix the two but still skew positional. Suggests `8.6` is the "first" S-Inhibition head the model learned and the others are refinements.

### Learning

- **The two-head induction circuit shows up in many places.** Once prev-token heads (`2.2`, `4.11`) are validated via path-patching to the induction heads' keys, the same induction mechanism the model learned for plain repeated-token sequences is provably being repurposed in IOI to detect "S is duplicated." Circuits compose.
- **Naive ablation underestimates effects when backups exist.** The right baseline isn't "remove head, measure delta"; it's "predict the delta from the head's direct contribution and see if the observed delta is smaller." A smaller-than-predicted delta is the signature of compensation.
- **`gen_flipped_prompts` is the right primitive for factorial interventions.** Decomposing a head's input into "what signal does it actually use?" requires generating prompts where each signal can be independently corrupted. The IOIDataset method exposes all 2^k corruption combinations as a single API.
- **Within a class of heads, individual heads can specialize.** `8.6` does the position-only job; `7.3`/`7.9`/`8.10` do mixed jobs. The IOI paper's clean 7-class taxonomy hides this fine-grained heterogeneity, and looking at it per-head changes the story.
- **Backup heads + Negative Name Mover sign-flips imply learned redundancy.** It's not just that the model has spare heads; some heads actively *change* what they do when a primary is ablated. This is a property of the trained model, not an artifact of the experiment.

---

## Takeaways

1. **Three-stage causal toolkit**: observational logit attribution (`cache.decompose_resid` + `apply_ln_to_stack` + dot with logit diff direction) -> activation patching (`patching.get_act_patch_*` to find which (layer, position) or (layer, head) slots matter) -> path patching (which specific edges, e.g. head -> downstream V input). Each is more rigorous than the last; each takes more compute.
2. **The IOI circuit is the canonical "early-mid-late" picture.** Duplicate-token + induction in early layers signal "S is duplicated"; S-Inhibition heads in mid layers (via K-composition with the first stage) write "attend to IO, not S" into END; Name Movers in late layers attend to IO and copy it. The same template generalizes to many circuits.
3. **Localization is the headline.** Almost all useful computation is at the **S2 position** (early-mid layers) and the **END position** (mid-late layers). Everywhere else is essentially a passthrough. This is what makes patching maps look clean.
4. **MLPs are mostly silent on token-movement tasks.** Except `MLP0`, which functions as an extended embedding in GPT-2-small. For attention-driven circuits, all the action is in attention heads.
5. **Path patching > activation patching for circuit edges.** When the downstream path matters (head -> head V composition, not head -> final residual), path patching isolates the specific causal connection, while activation patching conflates it with everything downstream.
6. **Apply LayerNorm scale before projecting onto logit-diff directions.** `cache.apply_ln_to_stack(stack, layer=-1, pos_slice=-1)` is non-negotiable for clean direct logit attribution. Without it, the per-component contributions don't sum to the actual logit diff.
7. **Redundancy / backup heads are real.** Single-head ablation undercounts importance. Accurately measuring a head's role requires controlling for compensation - either via path patching, or by measuring on a minimal circuit where backups are also ablated.
8. **`fold_ln`, `center_unembed`, `center_writing_weights`, `refactor_factored_attn_matrices`** - the four flags make per-component linear analysis exact rather than approximate. Use them whenever doing logit attribution on GPT-2 family models.
9. **The bonus section is where the clean story breaks.** Naive ablation underestimates importance (backup heads compensate); the early "duplicate-token" signal is actually doing induction (heads have multiple roles); S-Inhibition heads aren't a uniform class (`8.6` is purely positional, others mix). The lesson: a clean 7-class circuit is the first draft, not the final story.

---

## Mini-glossary

- **IO (Indirect Object)**: the "correct" name to complete with (e.g. `" Mary"` in `"John gave the bag to Mary"`).
- **S (Subject)**: the duplicated name (e.g. `" John"`). Appears at S1 and S2.
- **S1, S2, END**: the three key token positions. S1 = first occurrence of subject; S2 = second occurrence; END = position right before the answer.
- **Logit diff**: `logit[IO] - logit[S]` at the final position. The metric of the whole exercise. Equal to log-prob diff after softmax.
- **Logit diff direction**: `W_U[:, IO] - W_U[:, S]`. A vector in residual-stream space. A component's contribution to the logit diff is its projection onto this direction (after final LN scale).
- **Duplicate Token Head**: attends from S2 to S1; writes "S is duplicated" into the residual. Examples: `0.1`, `3.0`.
- **Induction Head**: attends to the token after a previous occurrence of the current token. Used here for the same "S is duplicated" signal. Examples: `5.5`, `6.9`.
- **S-Inhibition Head**: at END, reads the duplicate-S signal via Q/K composition with duplicate/induction heads, writes "attend to IO, suppress S" into END. Examples: `7.3`, `7.9`, `8.6`, `8.10`.
- **Name Mover Head**: at END, attends strongly to IO, OV-copies it into the residual in the IO direction. Examples: `9.6`, `9.9`, `10.0`.
- **Negative Name Mover Head**: same shape as Name Mover but with opposite sign. Examples: `10.7`, `11.10`.
- **Backup Name Mover Head**: dormant when primary Name Movers are intact; activates if they're ablated.
- **Activation patching**: replace one activation slot with its clean value during a corrupted forward pass; measure metric recovery. Causal but conflates direct + downstream effects.
- **Path patching**: replace one node's output during a forward pass, *freeze all alternative paths*, measure metric change. Isolates a single edge in the computation graph.
- **`ioi_metric`**: linear interpolation calibrated so corrupted -> 0, clean -> 1. (Section 4 flips the sign: clean -> 0, ABC -> -1.)
- **ABC-corrupted dataset**: like the standard corrupted dataset but with both names replaced by unrelated names. Used in path patching as the "destroyed performance" baseline.
- **Faithful / Complete / Minimal**: three properties a proposed circuit should have. Faithful = circuit alone is enough; Complete = nothing outside it is needed; Minimal = nothing inside it can be removed.
- **`refactor_factored_attn_matrices`**: a HookedTransformer flag that rotates `W_Q`/`W_K` to share a singular-value basis. Makes head-level interpretation cleaner.
