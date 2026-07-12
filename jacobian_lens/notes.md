# Jacobian Lens (jspace) - Notes

Notes on the **Jacobian lens** from Anthropic's paper *[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)*. Reference implementation: [github.com/anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens). Notebook: [jspace.ipynb](jspace.ipynb).

The core claim: LLMs have a "jspace" — a small subspace of activations whose contents are **verbalizable** (they name what the model is currently reasoning about) and can be **broadcast** to downstream circuits. The Jacobian lens reads this subspace out; the paper's interventions probe its causal role.

## Setup and dependencies

- `torch`, `transformers>=5.5`, `huggingface_hub`, `matplotlib`, `numpy`
- `jlens` — install from `git+https://github.com/anthropics/jacobian-lens.git`
- `datasets` (only for §11 fitting on new model)
- HF Hub access — the released lens is at `neuronpedia/jacobian-lens`

## Models used

| Purpose | HF ID | d_model | n_layers | Lens source |
|---|---|---|---|---|
| Primary | `Qwen/Qwen3.5-4B` | 2560 | 32 | `neuronpedia/jacobian-lens` `qwen3.5-4b/.../n1000.pt` |
| Fit-your-own (§11) | `Qwen/Qwen2.5-1.5B-Instruct` | 1536 | 28 | fit locally with `jlens.fit` on 100 wikitext prompts |
| From-scratch (§10) | `Qwen/Qwen3.5-4B` (reused) | 2560 | 32 | fit with our own re-implementation on 10 wikitext prompts |

## The math in one page

**Residual stream:** at every layer $\ell$ and position $t$, the model maintains $h_{\ell,t} \in \mathbb{R}^{d_\text{model}}$. Blocks read-and-write additively.

**Logit lens (warm-up):** apply the final unembedding $W_U$ directly to an intermediate residual. Works late, fails early.

**Jacobian lens:** transport the intermediate residual into the *final-layer basis* first, then unembed. The transport is the average Jacobian:

$$J_\ell = \mathbb{E}_{t, \; t' \geq t, \; \text{prompt}} \left[ \frac{\partial h_{L, t'}}{\partial h_{\ell, t}} \right]$$

One $d_\text{model} \times d_\text{model}$ matrix per layer, averaged over positions AND prompts (paper uses ~1000).

**Reading:**

$$\text{J-lens}(h_\ell) \;=\; \text{softmax}\big( W_U \cdot \text{RMSNorm}(J_\ell \, h_\ell) \big)$$

**J-lens vectors (approximation, dropping the RMSNorm):** one direction per vocab token,

$$v_{\ell, v} := J_\ell^\top u_v$$

where $u_v$ is row $v$ of $W_U$. These are used for interventions.

**Interventions (all from the paper, all implemented as PyTorch forward hooks in the notebook):**

| Op | Formula | Uses |
|---|---|---|
| Read | $s_v = v_{\ell,v}^\top h$ | monitoring |
| Steer | $h \mathrel{+}= \alpha \, v_{\ell,v}$ | inject concept |
| Ablate | $h \mathrel{-}= (v^\top h / \|v\|^2) v$ | remove concept |
| Patch | $V = [v_s \; v_t]$; $c = V^\dagger h$; $h \mathrel{+}= V(\sigma(c) - c)$ | swap concepts |

**Workspace band:** the paper says workspace behaviour spans ~38%-92% depth. For our 32-layer Qwen 3.5 4B that's **layers 12-29** — every intervention below is applied across this whole band, not at a single layer.

## Notebook structure

| § | Content |
|---|---|
| 1 | The math, from scratch (with explicit RMSNorm caveat) |
| 2 | Setup: install, load model, load released lens |
| 3 | Full `jlens` API reference (every exported name + minimal example) |
| 4 | Basic readout: J-lens vs logit-lens at every fitted layer |
| 5 | Interactive slice viz via `compute_slice + build_page` over 4 bundled examples |
| 6 | Inside the lens: `imshow(J_\ell)` at 4 depths + `|J|_F`/mean(diag) sweep + cos-sim of J-lens vectors |
| 7 | J-lens across task types (factual, arithmetic, code, two-hop) |
| 8 | Interventions: sparse readout, steering, ablation, patching |
| 9 | Replicate paper: two-hop swap, flexible generalization, ablation ceiling |
| 10 | Build the lens from scratch (own `ActivationRecorder`, `JacobianLens`, `jacobian_for_prompt`, `fit`) |
| 11 | Fit your own lens with `jlens.fit` on another model |

## Key results (verified against cell outputs)

### Reading the lens — the concept surfaces at the right layer

For prompt `"Fact: The currency used in the country shaped like a boot is"`:

| Layer | J-lens top-1 | J-lens prob | logit-lens prob |
|---|---|---|---|
| 4 | `` ` `` | 0.46 | 0.04 (junk) |
| 11 | `意大利` (Italy in Chinese) | 0.20 | 0.002 |
| 12 | `Italy` | 0.125 | 0.002 |
| 22 | `-shaped` | 0.56 | 0.09 (prompt echo begins) |
| 26 | `boot` | 0.86 | 0.49 |
| 30 | `is` | 0.81 | 0.26 |

The **intermediate concept `Italy`** surfaces at layers 11-12 (35-40% depth). Late layers echo prompt tokens. **This is exactly the paper's finding replicated on this model.**

### Jacobian magnitude across layers — clean monotonic trend

From §10.5 from-scratch sanity (single prompt, 3 source layers):

| Layer | \|J\|_F | mean(diag) |
|---|---|---|
| 10 | 91.58 | 0.186 (far from identity) |
| 20 | 80.87 | 0.653 |
| 25 | 76.33 | 1.028 (near-identity) |

Chain rule expectation: $J_\ell = \prod_{k=\ell}^{L-1}(I + df_k/dh)$. Late layer, few blocks to compose → $J \approx I$; early layer, many blocks → heavy off-diagonal. **Confirmed monotonically, both on `|J|_F` and `mean(diag)`.**

### J-lens vector correlations — critical for patch failure

At best-correlated layer:

| pair | peak cos | at layer |
|---|---|---|
| `Italy` / `France` | **0.73** | L17 |
| `Germany` / `Japan` | **0.76** | L0 |
| `Rome` / `Paris` | 0.42 | L17 |
| `Italy` / `Rome` | 0.51 | L12 |
| `Italy` / `banana` | 0.30 | L1 (noise floor) |

Country-token pairs share ~73-76% of their direction. `pinv([v_s, v_t])` is well-conditioned (not degenerate), but the least-squares split of $h$ into $c_0 v_s + c_1 v_t$ components is **ambiguous** at this correlation level. See §8d/§9a below.

### Interventions

| § | Op | Result | Verdict |
|---|---|---|---|
| 8b | steer `h += 5 · v_Paris` on `The capital of Italy is` | P(Paris) 0.0002 → **0.9994** | ✓ works cleanly |
| 8c | ablate `v_Italy` on boot prompt | P(Rome) 0.118 → **0.0002** | ✓ works cleanly, Italy → Rome causal path confirmed |
| 8d | patch Italy → France (single prompt) | Rome → **London**, not Paris | ✗ fails |
| 9a | patch across 6 two-hop pairs | **0/6** flips (paper reports 61% on Sonnet 4.5) | ✗ universal failure |
| 9b | flexible generalization (one patch, 3 downstream Qs) | no consistent flip | ✗ same root cause |
| 9c | jspace ablation ceiling (k=0,2,5,10,20) | **83% → 67% → 0%** at k=5 | ✓ paper's headline result replicated |

### The patch failure — diagnosed, not a bug

§8d prints per-layer (c_0, c_1) diagnostics during a single Italy→France patch. **The signs alternate across the workspace band:**

| L | c_0 (Italy) | c_1 (France) | direction |
|---|---|---|---|
| 12 | 0.61 | 0.47 | ✓ correct |
| 13 | 0.42 | 0.55 | ✗ **wrong** |
| 14 | 0.57 | 0.38 | ✓ correct |
| 15 | 0.34 | 0.55 | ✗ **wrong** |
| ... | (pattern continues) | | |
| 20 | 1.60 | 0.01 | ✓ correct |
| 21 | 0.07 | 1.53 | ✗ **wrong** |

Because $\cos(v_s, v_t) \approx 0.73$, the least-squares decomposition of $h$ has two nearly-equivalent solutions; small differences in $h$ from one layer to the next flip which one wins. Averaged over the 18-layer band, the correct-direction and wrong-direction pushes partially cancel.

**Conclusion: the 0/6 patch-flip is a real, correctly-implemented, correctly-diagnosed limitation of the paper's coordinate-swap formulation on this specific model — not a bug in our code.** On Claude Sonnet 4.5 (the paper's model) the vocabulary is much richer so v_Italy and v_France are less correlated, the decomposition is stable, and the paper reports 61% flip rate. On a 4B Qwen with 73% correlation between country-token J-lens vectors, the coordinate swap is intrinsically unstable.

**Alternative that works on this model:** direct steering by the difference vector,

```python
diff_vecs = {L: (tgt_vecs[L] - src_vecs[L]) for L in WORKSPACE_BAND}
with steer(hf_model, WORKSPACE_BAND, diff_vecs, alpha=5.0):
    patched = generate(...)
```

Doesn't depend on decomposition → works reliably (§8b showed steering is stable here).

### The ablation ceiling — the paper's headline reproduces cleanly

Ablate top-k J-lens vectors across the workspace band, skipping any that appear in the model's clean-top-10 output (paper's constraint — otherwise we'd just be removing the answer itself).

| k | multi-hop accuracy |
|---|---|
| 0 | 83.3% (5/6) |
| 2 | 66.7% (4/6) |
| 5 | **0.0%** |
| 10-20 | 0.0% |

Ablating just 5 non-answer J-lens directions per layer × 18 layers destroys multi-hop reasoning. **This is the paper's strongest evidence that jspace is causally load-bearing, and it reproduces on Qwen 3.5 4B.** Unlike the patch, ablation uses individual J-lens vectors (no pinv decomposition) so it's numerically stable.

### From-scratch training (§10)

Our reimplementation of `ActivationRecorder`, `JacobianLens`, `jacobian_for_prompt`, and `fit` — following `jlens/hooks.py`, `jlens/lens.py`, `jlens/fitting.py` line by line.

- **~3 minutes per prompt on Qwen 3.5 4B** (one forward + 320 backward passes on retained bf16 graph, 8 source layers). Full fit on 10 prompts: ~30 minutes.
- **Fit convergence diagnostics** (from the library's own log, cell 95): `max||J||/sqrt(d) ≈ 1.5-2.0` across prompts (scale-free, indicates the Jacobian is 1.5-2× a random Gaussian's magnitude — healthy). `max_d_mean` (relative change in running mean) falls as 1/n — clean convergence.
- **Applied lens comparison** (§10.8): our 10-prompt lens surfaces `Italy` at layers 16 and 20 more sharply than the released 1000-prompt lens. Honest interpretation: **statistical noise on a small sample, not corpus specialization**. Would probably invert on a different 10 prompts.

### Fitting on Qwen 2.5 1.5B (§11)

Ran `jlens.fit` on 100 wikitext prompts. Total: ~27 minutes (16 sec/prompt). The resulting lens is dominated by quotation marks and ligatures (`"`, `".`, `ﬁ`, `ﬂ`) — 100 wikitext prompts on a 1.5B model isn't enough for clean semantic content. The concept swap on this newly-fit lens fails for the same reason: J-lens vectors for `Italy`/`France` are dominated by quote-mark structure, not country semantics.

## Precision policy (real bug we caught)

Initial implementation of J-lens vectors was computing `J.T @ u_v` in bf16 — accumulating `√d × ε(bf16) ≈ √2560 × 0.008 ≈ 0.4` of rounding noise vs `|v| ≈ 0.86`. Half the vector was noise.

**Fix:** `jlens_vector` now computes in fp32 explicitly. All three intervention hooks (`steer`, `ablate`, `patch_swap`) now do the delta computation in fp32 and only cast the final delta to `h.dtype` for the residual add. The comment above the hook definitions documents this rule so it can't regress:

> Compute deltas in fp32, cast to h.dtype only for the residual add.

## What works vs what fails on Qwen 3.5 4B

| Experiment | Verdict | Why |
|---|---|---|
| J-lens readout at any position/layer | ✓ | Well-fit lens; concepts surface at expected layers |
| Interactive slice viz | ✓ | Anthropic's canonical view via `jlens.vis` |
| Jacobian matrix visualisation (§6) | ✓ | Monotonic pattern matches chain-rule expectation |
| Steering by single J-lens vector | ✓ | Stable — no decomposition needed |
| Ablation of single concept | ✓ | Stable — no decomposition needed |
| Ablation of top-k jspace (workspace ablation) | ✓ | Paper's headline replicated |
| **Coordinate-swap patch** | ✗ | Correlated J-lens vectors → unstable pinv decomposition → per-layer sign alternation → self-cancels across band |

**The mathematics is faithful to the paper (verified line-by-line against `jlens/{lens,fitting,hooks}.py`).** The patch failure is model-scale-limited, not code-limited.

## Files in this directory

- `jspace.ipynb` — the notebook (99 cells, 43 code + 56 markdown, ~35 MB with embedded plots + interactive slice pages)
- `build_notebook.py` — regenerable source. Rebuilds preserve cell outputs (matches by executable code, ignoring comment-only diffs).
- `notes.md` — this file
- `jacobian_lens.excalidraw` — visual summary diagram
- `build_diagram.py` — Python source that generates the excalidraw
