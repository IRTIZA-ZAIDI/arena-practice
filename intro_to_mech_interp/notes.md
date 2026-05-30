# Intro to Mech Interp - Notes

Notes from ARENA [1.2] Intro to Mech Interp (TransformerLens + induction circuits). Companion to [1_2_Intro_to_Mech_Interp_exercises.ipynb](1_2_Intro_to_Mech_Interp_exercises.ipynb).

The whole notebook is built around **induction heads** as the running example. I started by loading a real GPT-2 with TransformerLens, then moved to a toy attention-only 2-layer model where the full induction circuit can actually be reverse-engineered. Hooks, ablations, logit attribution, and composition-score analysis all show up in service of that one circuit.

Reference model for the second half: `callummcdougall/attn_only_2L_half` (2 layers, attention-only, no LayerNorm, positional embeddings only fed to Q/K - chosen so the residual stream stays interpretable).

---

## Big picture (what mech interp *is*, operationally)

- **Privileged bases** are the few places where the model's representations are interpretable: the input tokens, the output logits, and the attention patterns. The residual stream, K/Q/V vectors, and MLP neurons are all in arbitrary bases (rotated by arbitrary orthogonal matrices), so individual coordinates mean nothing on their own.
- A **circuit** is a path through the model that implements a specific behavior. For induction, the path is: previous-token head in L0 (writes "what token came before me" into the residual stream) -> induction head in L1 (uses that to attend to the token *after* the previous occurrence of the current token, then copies it).
- A transformer block can be decomposed into 4 sub-circuits per head:
  - **QK circuit** (`W_Q @ W_K.T`): determines *which* source token to attend to.
  - **OV circuit** (`W_V @ W_O`): determines *what to write* once attention has been paid.
- Heads compose with each other across layers in three ways:
  - **Q composition**: layer-1 head's queries depend on layer-0 head's output.
  - **K composition**: layer-1 head's keys depend on layer-0 head's output (this is what powers induction).
  - **V composition**: layer-1 head's values depend on layer-0 head's output.

---

## 1. TransformerLens: introduction

### Tasks
- Loaded `gpt2-small` via `HookedTransformer.from_pretrained("gpt2-small")`.
- Inspected `gpt2_small.cfg` for `n_layers=12`, `n_heads=12`, `n_ctx=1024`.
- Ran the model and counted how many tokens the model predicted correctly (`argmax` then compare against the shifted ground truth).
- Cached all activations with `run_with_cache(tokens, remove_batch_dim=True)`. The cache supports two indexing styles: shorthand (`cache["pattern", 0]`) and full hook name (`cache["blocks.0.attn.hook_pattern"]`).
- **Manually reconstructed the attention pattern** from cached Q/K to verify understanding of the QK math (dot product, scale by `sqrt(d_head)`, mask, softmax).
- Visualized attention with `cv.attention.attention_patterns(tokens=..., attention=...)`.

```python
# Manually reconstruct attention pattern from cached Q and K.
q, k = gpt2_cache["q", 0], gpt2_cache["k", 0]   # both [seq, n_head, d_head]
seq, nhead, headsize = q.shape

attn_scores = einops.einsum(q, k, "seqQ n h, seqK n h -> n seqQ seqK")
mask = t.triu(t.ones((seq, seq), dtype=t.bool), diagonal=1).to(device)
attn_scores.masked_fill_(mask, -1e9)
pattern_from_qk = (attn_scores / headsize**0.5).softmax(-1)

t.testing.assert_close(gpt2_cache["pattern", 0], pattern_from_qk)
```

### Learning
- TransformerLens normalizes the parameter layout across architectures (`W_Q`, `W_K`, `W_V`, `W_O` are kept separate per head, with consistent shapes like `[n_layers, n_heads, d_model, d_head]`). Once these conventions are internalized, every model in the library looks the same.
- The cache indexing shorthand (`cache["pattern", 0]`) calls `utils.get_act_name("pattern", 0)` under the hood. Use the shorthand by default; fall back to the full string only when the cache doesn't recognize the shortname.
- **The off-by-one is everywhere** (again): predictions at position `s` are scored against `tokens[s+1]`. `correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")` is the clean version.

---

## 2. Finding induction heads

### Tasks
- Loaded the **toy 2L attention-only model** (no LayerNorm, no MLP, positional embeddings added only to Q/K inputs). All the circuit analysis lives on this model because removing LayerNorm + MLPs makes everything linear and decomposable.
- Visualized attention patterns for both layers; spotted three basic patterns by eye:
  - **Current-token heads** (attention to the diagonal).
  - **Previous-token heads** (attention one off the diagonal).
  - **First-token heads** (attention to the BOS column).
- Wrote three detectors that score each head by mean attention on the relevant slice of the matrix.
- Generated **repeated random-token sequences** `[BOS, *rand, *rand]` to probe in-context learning. A model with induction heads will get low loss on the *second* copy even though it's never seen this sequence in training.
- Computed per-token log-probs on the repeated sequence and saw the loss drop sharply on the second half.
- Wrote an **induction-head detector**: average attention on the diagonal offset by `-(seq_len - 1)`. Destination token `T` at position `seq_len + 1 + i` attends back to position `1 + i + 1` (one *after* the first occurrence of `T`).

```python
def current_attn_detector(cache):
    """Heads attending to themselves (the main diagonal)."""
    heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            pattern = cache["pattern", layer][head]
            if pattern.diagonal().mean() > 0.4:
                heads.append(f"{layer}.{head}")
    return heads


def prev_attn_detector(cache):
    """Heads attending to position-1 (offset=-1 = one below the main diagonal)."""
    heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            pattern = cache["pattern", layer][head]
            if pattern.diagonal(offset=-1).mean() > 0.4:
                heads.append(f"{layer}.{head}")
    return heads


def first_attn_detector(cache):
    """Heads attending to the first token (the BOS column)."""
    heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            pattern = cache["pattern", layer][head]
            if pattern[:, 0].mean() > 0.4:
                heads.append(f"{layer}.{head}")
    return heads


def generate_repeated_tokens(model, seq_len, batch_size=1):
    """[BOS, *rand, *rand] - the canonical induction probe."""
    t.manual_seed(0)
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    half = t.randint(low=0, high=model.cfg.d_vocab, size=(batch_size, seq_len), dtype=t.int64)
    return t.cat([prefix, half, half], dim=-1).to(device)


def induction_attn_detector(cache):
    """Heads attending to the induction stripe (diagonal at offset -(seq_len-1))."""
    heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            pattern = cache["pattern", layer][head]
            seq_len = (pattern.shape[-1] - 1) // 2
            if pattern.diagonal(-seq_len + 1).mean() > 0.4:
                heads.append(f"{layer}.{head}")
    return heads
```

### Results
- For the toy model, detectors identified clear current/prev/first-token heads in layer 0, and induction heads `1.4` and `1.10` (with `1.6` borderline) in layer 1.
- Loss on repeated random sequences: high on the first half (unpredictable), low on the second half. The drop is *the* signature of an induction circuit.

### Learning
- `t.diagonal(offset=k)` is the operation that turns "attention to relative position" questions into single-tensor reductions. Positive offset = above diagonal (future, which is masked), negative offset = below (past).
- The induction-stripe offset is `-(seq_len - 1)`, not `-seq_len`. The +1 is because the destination token attends to the token **after** the previous occurrence of itself.
- The repeated-tokens probe is the cleanest test for in-context learning because the model has provably never seen this sequence in training - any predictive accuracy on the second copy is generalization.

---

## 3. TransformerLens: hooks, ablation, logit attribution

This section moved from observation (looking at patterns) to **causal intervention** (changing activations and measuring the effect).

### Tasks
- Learned the `model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])` API. Each hook function takes `(activation, hook)` and can either modify the activation in place or just read it.
- Wrote an **induction-score hook** that pulls out the induction stripe from each layer's attention pattern and writes it to a global tensor (parallel to the manual detector above, but via the hook API).
- Ran the same induction-score analysis on GPT-2 small. Several middle-layer heads (`5.x`, `6.x`, `7.x`) score high.
- Implemented **logit attribution**: decompose each correct-token logit into contributions from the direct path (embed -> unembed), L0 heads, and L1 heads.
- Implemented **head ablation** (zero and mean variants): set head `h`'s output to zero (or its batch mean) and measure the loss increase on the second half of repeated sequences.

```python
# Hook that aggregates induction scores across all attention pattern hooks.
induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=device)

def induction_score_hook(pattern, hook):
    """pattern: (batch, head, q_pos, k_pos). Writes one score per head."""
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)
    score = einops.reduce(induction_stripe, "batch head position -> head", "mean")
    induction_score_store[hook.layer(), :] = score

# Filter selects only attention pattern hooks (one per layer).
pattern_hook_names_filter = lambda name: name.endswith("pattern")
model.run_with_hooks(
    rep_tokens_10,
    return_type=None,  # don't compute logits, saves compute
    fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
)


def logit_attribution(embed, l1_results, l2_results, W_U, tokens):
    """Decompose each correct next-token logit into direct + per-head contributions.

    Returns (seq-1, 1 + 2*n_heads).
    """
    # Select the unembed column for the *correct* next token at each position.
    W_U_correct = W_U[:, tokens[1:]]                                     # (d_model, seq-1)

    direct = einops.einsum(W_U_correct, embed[:-1],
                           "d seq, seq d -> seq").unsqueeze(-1)          # (seq-1, 1)
    l0 = einops.einsum(W_U_correct, l1_results[:-1],
                       "d seq, seq h d -> seq h")                        # (seq-1, n_heads)
    l1 = einops.einsum(W_U_correct, l2_results[:-1],
                       "d seq, seq h d -> seq h")                        # (seq-1, n_heads)
    return t.cat([direct, l0, l1], dim=-1)


def head_zero_ablation_hook(z, hook, head_index_to_ablate):
    """Zero out the output of one head. z: (batch, seq, n_heads, d_head)."""
    z[:, :, head_index_to_ablate, :] = 0


def head_mean_ablation_hook(z, hook, head_index_to_ablate):
    """Replace with batch mean - keeps the model closer to its training distribution."""
    z[:, :, head_index_to_ablate, :] = z[:, :, head_index_to_ablate, :].mean(0)


def get_ablation_scores(model, tokens, ablation_function=head_zero_ablation_hook):
    """For each head, ablate it and report the loss increase on the second half."""
    scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=device)
    model.reset_hooks()
    seq_len = (tokens.shape[1] - 1) // 2
    logits = model(tokens, return_type="logits")
    loss_clean = -get_log_probs(logits, tokens)[:, -(seq_len - 1):].mean()

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            hook_fn = functools.partial(ablation_function, head_index_to_ablate=head)
            # Ablate at the "z" hook (per-head per-position output before W_O).
            ablated_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(utils.get_act_name("z", layer), hook_fn)])
            loss = -get_log_probs(ablated_logits, tokens)[:, -(seq_len - 1):].mean()
            scores[layer, head] = loss - loss_clean
    return scores
```

### Results
- **Logit attribution** on the repeated sequence: first half is noise (no predictable structure), second half shows large contributions from heads `1.4` and `1.10`. The direct path also contributes a lot, but mostly at "boring" tokens (BOS, common punctuation) where the unigram prior dominates.
- **Ablation** on the repeated sequence: zeroing head `0.7` collapses induction performance (it's the previous-token head). Zeroing `1.4` or `1.10` also hurts a lot. Mean ablation is cleaner than zero ablation - zeroing pushes the model far off its training distribution and produces noisier downstream effects.

### Learning
- **Ablation and logit attribution give different kinds of evidence.** Logit attribution says "this head writes something useful to the residual stream." Ablation says "removing this head breaks the model." A head with high logit attribution but zero ablation effect is being compensated for by another head; a head with low logit attribution but big ablation effect is upstream in the circuit.
- **Mean-ablate, not zero-ablate, for realism.** Zeroing takes the model off-distribution; mean-ablation only removes the per-batch-item information while keeping the activation roughly in-distribution.
- Use `utils.get_act_name("z", layer)` rather than hand-writing hook strings. The `z` hook fires after V is applied per head and before `W_O` mixes them - that's the right place to ablate a single head's contribution.

---

## 4. Reverse-engineering the induction circuit

This is where the toy model earns its keep. With LayerNorm and MLPs removed, every path through the model is a product of weight matrices, and I could just multiply them out and stare at them.

### Tasks
- Computed the **OV circuit** for head `1.4`: `W_E @ W_V[1,4] @ W_O[1,4] @ W_U`, kept as a `FactoredMatrix` so we never materialize the `(d_vocab, d_vocab)` product.
- Visualized a 200-row submatrix sample - looks vaguely diagonal-ish (the copying signature).
- Computed **top-1 accuracy** (fraction of rows whose argmax is the diagonal): ~30% for head `1.4` alone, ~95% for the **effective OV circuit** combining heads `1.4` and `1.10`. The pair has rank 128 vs each head's rank 64, which is enough to approximate the identity over a 50k vocab.
- Computed the **QK prev-token circuit** for head `0.7`: `W_pos @ W_Q[0,7] @ W_K[0,7].T @ W_pos.T`, then masked + scaled + softmaxed. The result has high values just below the diagonal - that's literally the previous-token attention pattern, derived from weights alone.
- Decomposed the **QK input** to layer 1 into 14 components (token embed + pos embed + 12 layer-0 head outputs). Each component independently projects through `W_Q[1,4]` and `W_K[1,4]` to give per-component q and k vectors. Then computed the `[q_component, k_component, q_pos, k_pos]` decomposed attention scores.
- Confirmed via the decomposition that the dominant pair is **(query = token embed, key = output of head 0.7)** - the K-composition signature.
- Computed the **full K-composition circuit**: `W_E @ W_Q[1,4] @ W_K[1,4].T @ W_O[0,7].T @ W_V[0,7].T @ W_E.T`. Stored as `FactoredMatrix(W_E @ W_Q, (W_E @ W_V @ W_O @ W_K).T)`.
- Computed **composition scores** (Frobenius-norm ratio) for all pairs of L0 and L1 heads, in Q/K/V variants. Established a baseline by drawing 200 Kaiming-initialized random pairs and taking the mean.
- Ran **targeted ablations** (`ablation_induction_score`): ablate one L0 head, then measure how the induction score of L1 head `1.4` (or `1.10`) changes.

```python
# --- OV circuit for head 1.4 ---
head_index, layer = 4, 1
W_E, W_U = model.W_E, model.W_U
W_V, W_O = model.W_V[layer, head_index], model.W_O[layer, head_index]
# Kept as a FactoredMatrix: never materialize the (d_vocab, d_vocab) product.
full_OV_circuit = FactoredMatrix(W_E @ W_V, W_O @ W_U)


def top_1_acc(full_OV_circuit, batch_size=1000):
    """Fraction of rows whose argmax is the diagonal (the copying signature)."""
    total, vocab = 0, full_OV_circuit.shape[0]
    for start in range(0, vocab, batch_size):
        end = min(start + batch_size, vocab)
        block = full_OV_circuit[start:end].AB                 # materialize a slice only
        max_idx = block.argmax(dim=1)
        diag_idx = t.arange(start, end, device=block.device)
        total += (max_idx == diag_idx).float().sum().item()
    return total / vocab


# --- Effective OV circuit: heads 1.4 and 1.10 together (rank 128) ---
W_O_both = einops.rearrange(model.W_O[1, [4, 10]], "h d_h d_m -> (h d_h) d_m")
W_V_both = einops.rearrange(model.W_V[1, [4, 10]], "h d_m d_h -> d_m (h d_h)")
W_OV_eff = W_E @ FactoredMatrix(W_V_both, W_O_both) @ W_U


# --- QK prev-token circuit for head 0.7 (positional-only) ---
W_pos = model.W_pos
W_QK = model.W_Q[0, 7] @ model.W_K[0, 7].T
pos_by_pos_scores = W_pos @ W_QK @ W_pos.T
mask = t.tril(t.ones_like(pos_by_pos_scores)).bool()
pos_by_pos_pattern = t.where(mask,
                             pos_by_pos_scores / model.cfg.d_head ** 0.5,
                             -1.0e6).softmax(-1)


# --- Decompose the input to layer 1 attention into 14 sources ---
def decompose_qk_input(cache):
    """Return shape (n_heads+2, posn, d_model): [embed, pos_embed, *L0 head outputs]."""
    embed = cache["embed"].unsqueeze(0)
    pos_embed = cache["pos_embed"].unsqueeze(0)
    l0_results = einops.rearrange(cache["result", 0], "p h d -> h p d")
    return t.cat([embed, pos_embed, l0_results], dim=0)


def decompose_attn_scores(decomposed_q, decomposed_k, model):
    """[q_comp, k_comp, q_pos, k_pos] - score contribution from each (q_src, k_src) pair."""
    return einops.einsum(
        decomposed_q, decomposed_k,
        "q_comp q_pos d, k_comp k_pos d -> q_comp k_comp q_pos k_pos"
    ) / (model.cfg.d_head ** 0.5)


# --- K-composition full circuit: W_E @ W_QK[1.4] @ W_OV[0.7].T @ W_E.T ---
def find_K_comp_full_circuit(model, prev_token_head_index, ind_head_index):
    W_E = model.W_E
    W_Q, W_K = model.W_Q[1, ind_head_index], model.W_K[1, ind_head_index]
    W_O, W_V = model.W_O[0, prev_token_head_index], model.W_V[0, prev_token_head_index]
    Q = W_E @ W_Q
    K = W_E @ W_V @ W_O @ W_K
    return FactoredMatrix(Q, K.T)


# --- Composition score: how much of W_A's output survives projection through W_B ---
def get_comp_score(W_A, W_B):
    """||W_A W_B||_F / (||W_A||_F ||W_B||_F)."""
    W_A_norm = W_A.pow(2).sum().sqrt()
    W_B_norm = W_B.pow(2).sum().sqrt()
    W_AB_norm = (W_A @ W_B).pow(2).sum().sqrt()
    return (W_AB_norm / (W_A_norm * W_B_norm)).item()


# Fill in a (12, 12) composition matrix for each of Q/K/V composition.
W_QK = model.W_Q @ model.W_K.transpose(-1, -2)    # (n_layers, n_heads, d_model, d_model)
W_OV = model.W_V @ model.W_O                       # (n_layers, n_heads, d_model, d_model)
composition_scores = {k: t.zeros(model.cfg.n_heads, model.cfg.n_heads, device=device)
                      for k in ("Q", "K", "V")}
for i in range(model.cfg.n_heads):
    for j in range(model.cfg.n_heads):
        composition_scores["Q"][i, j] = get_comp_score(W_OV[0, i], W_QK[1, j])
        composition_scores["K"][i, j] = get_comp_score(W_OV[0, i], W_QK[1, j].T)
        composition_scores["V"][i, j] = get_comp_score(W_OV[0, i], W_OV[1, j])


# --- Targeted ablation: how much does L1 head's induction depend on each L0 head? ---
def ablation_induction_score(prev_head_index, ind_head_index):
    def ablation_hook(v, hook):
        if prev_head_index is not None:
            v[:, :, prev_head_index] = 0.0
        return v

    def induction_pattern_hook(attn, hook):
        hook.ctx["induction_score"] = attn[0, ind_head_index].diagonal(
            dim1=-2, dim2=-1, offset=1 - seq_len).mean()

    model.run_with_hooks(
        rep_tokens,
        return_type=None,
        fwd_hooks=[
            (utils.get_act_name("v", 0), ablation_hook),
            (utils.get_act_name("pattern", 1), induction_pattern_hook),
        ],
    )
    return model.blocks[1].attn.hook_pattern.ctx["induction_score"]
```

### Results
- **OV circuit (single head)**: head `1.4` alone gets ~30% top-1 (copying with low accuracy).
- **OV circuit (effective, 1.4 + 1.10)**: ~95% top-1 (copying with near-perfect accuracy). The two heads share the work because each is rank-64 and together they get rank-128, much closer to the rank needed to approximate identity over a 50k vocab.
- **QK prev-token circuit** for `0.7`: the softmaxed positional matrix has high values just below the diagonal - the previous-token pattern emerges directly from the weights, with no input dependence.
- **QK decomposition for `1.4`**: the (token-embed query, head-0.7 key) pair dominates. Every other (q_component, k_component) pair has tiny contribution. This is the signature of K-composition with `0.7`.
- **Composition scores**: K-composition between `0.7` and `1.4`/`1.10` is much higher than the random baseline (clear bright spots). V-composition for the same pair is *low* (the induction heads use `0.7` for *attention*, not for *what to copy*).
- **Targeted ablation**: ablating `0.7` drops the induction score for `1.4` to nearly zero. Ablating any other L0 head barely moves it. Conclusive evidence that `0.7` is the prev-token head feeding the induction heads.

### Learning
- **`FactoredMatrix(A, B)` is essential.** Many of the circuit matrices are `(d_vocab, d_vocab)` = 50k x 50k, which is 10GB at float32. Storing them as `A @ B` with a narrow middle dim lets norms, SVDs, and indexed slices be computed without ever materializing the product.
- **OV vs QK are conceptually independent.** OV asks "given that I attend, what do I copy?" QK asks "where do I attend?" The induction algorithm needs both: QK provides the offset-diagonal stripe, OV provides the copy.
- **Effective circuits matter.** Single heads can look weak (30% top-1) while pairs of heads doing the same thing are strong (95%). Always check whether multiple heads are doing the same job in parallel before declaring a head "weak."
- **The composition-score baseline matters more than the absolute number.** Frobenius ratios are not calibrated by themselves; the random-Kaiming baseline (~0.1 typically) is what tells "significantly above zero" from noise.
- **K-composition is the diagnostic for induction.** Seeing L0 head A and L1 head B with a high K-composition score, where A has a previous-token pattern, is strong evidence that B is an induction head. This generalizes beyond the toy model.

---

## Takeaways

1. **TransformerLens is just a thin layer over a standard transformer.** The win is consistent parameter shapes (`W_Q[layer, head, d_model, d_head]`), a clean hook API, and the `ActivationCache`. Once the conventions are internalized, the same analysis code reuses across all supported models.
2. **Privileged bases anchor interpretation.** Tokens (input), logits (output), and attention patterns are interpretable. Everything else is in an arbitrary basis - useful as intermediate state, but individual coordinates aren't readable.
3. **Three kinds of evidence converge in mech interp**:
   - **Observation** (attention patterns, activation visualizations): cheap, suggestive, not causal.
   - **Logit attribution / decomposition** (additive): says what each component contributes to a specific output.
   - **Ablation** (causal): says what breaks when a component is removed.
   - A confident claim about a circuit usually wants all three pointing the same way.
4. **Decompose along the QK/OV split.** Heads have two independent jobs - choosing where to look and choosing what to copy - and the math factorizes cleanly. Analyzing them together conflates them.
5. **Multi-head sums approximate full-rank matrices.** Each head is rank `d_head` = 64; the effective circuit summing multiple heads can be much higher rank and much closer to identity over `d_vocab`. This is why models split copying across multiple heads.
6. **Composition scores + baseline.** The Frobenius ratio of `||AB||_F / (||A||_F ||B||_F)` is a cheap way to scan for which heads talk to which - but always check against the random-init baseline.
7. **Repeated random tokens are the cleanest in-context probe.** Out-of-distribution by construction; any predictive accuracy is real generalization, not memorization.

---

## Mini-glossary

- **Induction head**: an L1+ attention head that, on token `T`, attends to the token *immediately after* a previous occurrence of `T`, and copies that token's content into the output.
- **Previous-token head**: an L0 attention head whose pattern is the diagonal offset by -1; it writes "what token came before me" into the residual stream.
- **OV circuit**: `W_V @ W_O` for a head. Determines what gets copied once attention is paid.
- **QK circuit**: `W_Q @ W_K.T` for a head. Determines where attention is paid.
- **Q/K/V composition**: layer-1 head reads its Q (or K or V) input from a residual-stream contribution written by a layer-0 head.
- **K-composition (the induction signature)**: `W_E -> W_OV[L0] -> W_QK[L1] <- W_E`. The induction head's keys depend on the previous-token head's writes.
- **Effective circuit**: sum of OV products across multiple heads that do the same job; higher rank than any individual head, often dramatically more accurate.
- **`FactoredMatrix`**: a lazy product of two matrices, exposing norms / SVDs / indexed slices without materializing the product. Essential for `(d_vocab, d_vocab)`-shaped circuits.
- **`run_with_hooks(tokens, fwd_hooks=[...])`**: TransformerLens's primary intervention API. Each hook is `(activation, hook_point) -> activation_or_None`.
- **`utils.get_act_name(name, layer)`**: maps short activation names ("pattern", "z", "v", "result", "embed", "pos_embed") to full module paths.
- **Composition score**: `||W_A @ W_B||_F / (||W_A||_F * ||W_B||_F)`. A scalar measure of how much of W_A's output direction survives projection through W_B.
- **Logit attribution**: per-position, per-component decomposition of the correct-next-token logit. Components are the direct path + each attention head per layer.
- **Zero ablation vs mean ablation**: zeroing takes the model off-distribution; mean ablation (substitute the per-batch mean) keeps it closer to its trained behavior.
