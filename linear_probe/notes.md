# Linear Probes - Notes

Notes from ARENA [1.3.1] Linear Probes. Companion to [1_3_1_Linear_Probes_exercises.ipynb](1_3_1_Linear_Probes_exercises.ipynb).

Replicates pieces of three papers:
- **Geometry of Truth** (Marks & Tegmark, 2023) - truth is linearly represented in LLM activations.
- **Deception Detection w/ Linear Probes** (Apollo Research, 2025) - same toolkit, applied to strategic deception.
- **Detecting High-Stakes Interactions with Activation Probes** (NeurIPS 2025) - attention-pooled probes on full-sequence activations.

Models used: `Llama-2-13b-hf` (base, for truth) and `Meta-Llama-3.1-8B-Instruct` (for deception + high-stakes).

---

## Big picture (what a linear probe *is*)

A linear probe is a small classifier (a single direction `w` in activation space, optionally with normalization) that you train on top of a frozen LLM's hidden states. The hypothesis being tested: **some property `P` (truth, deception, high-stakes) is encoded as a roughly-linear direction in the residual stream.** If a simple linear classifier on hidden states predicts `P` well, that's evidence the model represents `P` linearly.

Two things to keep distinct throughout:
1. **Classification accuracy** - does the probe predict the label?
2. **Causal relevance** - does intervening on that direction actually change model behavior? Accuracy alone is not enough; you need interventions.

---

## 1. Setup & visualizing truth representations

### What I did
- Implemented `extract_activations(statements, model, tokenizer, layers)`:
  - Tokenize with right-padding, run forward pass with `output_hidden_states=True`.
  - For each statement, pull the hidden state at the **last real token** (use `attention_mask.sum(-1) - 1` per row, not just `[:, -1]` - padding lives on the right).
  - Returns `dict[layer_idx -> (n, d_model)]`.
- Probed at `PROBE_LAYER = 14` (the value used in the Geometry-of-Truth config for Llama-2-13B).
- Datasets: `cities`, `sp_en_trans`, `larger_than` - short declarative true/false statements.
- Implemented `get_pca_components` via eigendecomposition of the centered covariance matrix (top-k eigenvectors, sorted by eigenvalue desc).
- Projected activations onto top-2 PCs and plotted, colored by label.
- Implemented `layer_sweep_accuracy`: at every layer, train a **difference-of-means classifier** (direction = `mean(true) - mean(false)`, decide by sign of dot product), report train/test accuracy.

### Results
- **PCA plots are visibly separable by true/false** for all three datasets, even though PCA is unsupervised. Truth is one of the *most prominent* directions of variance in mid-layer activations.
- **Layer sweep** peaks roughly in layers 8–20 for Llama-2-13B. Layer 14 is near the peak. Very early layers (0–5) and very late layers (35+) probe poorly - early layers haven't computed truth-relevant features yet; late layers have specialized for next-token prediction.

### Learning
- The last-real-token activation is what carries the model's "judgment" of a declarative statement. The end-of-sentence period token in particular aggregates a lot.
- Truth lives **mid-network**, not at the output. This generalizes - for most semantic probes, sweep the layers.
- PCA separation is the cleanest evidence that something is *linearly* represented and *prominent*; if the supervised probe works but PCA doesn't separate, the feature is linear but minor.

---

## 2. Training & comparing probes

### What I did
- 80/20 train/test split per dataset (seed 42).
- Implemented **`MMProbe`** (mass-mean / difference-of-means):
  - `direction = mean(true) - mean(false)`, stored as non-trainable `nn.Parameter`.
  - Optional pooled within-class covariance with pseudo-inverse for an IID-corrected variant.
  - Predict via sign of `(x · direction)`.
- Implemented **`LRProbe`** (logistic regression):
  - `nn.Sequential(Linear(d, 1, bias=False), Sigmoid())`.
  - `StandardScaler` mean/scale registered as buffers, applied at inference.
  - Trained via sklearn's `LogisticRegression`, then weights copied into the torch module.
- Built a **3×3 cross-dataset generalization matrix**: train probe on dataset i, evaluate on dataset j, for both MM and LR.

### Results
- In-distribution: MM and LR get **very similar accuracies** on all three datasets (both high).
- Cross-dataset: LR generalizes well at 13B (off-diagonal entries stay high) → evidence for a *unified* truth direction at this scale. MM also generalizes but slightly less than LR for raw classification.
- At smaller scales the paper reports more dataset-specific representations; we wouldn't see clean off-diagonals at, say, 7B.

### CCS (Contrastive Consistent Search) - discussion only
- Unsupervised: needs paired statements (`x`, `not x`), no labels. Loss enforces `p(x) ≈ 1 - p(neg_x)` (consistency) + confidence.
- The seductive claim was that truth could be discovered without labels - later work (Farquhar et al.) showed CCS often finds **prominent features that aren't truth**, just whatever varies most consistently between paired statements.
- Lesson: an unsupervised "probe" can succeed at its objective while missing the target concept entirely.

### Learning
- **Same accuracy ≠ same direction.** Two probes can hit 95% acc with directions that differ noticeably in cosine similarity. The interesting question is which direction the *model* actually uses - answered by interventions, not by accuracy.

---

## 3. Causal interventions

### What I did
- Implemented `few_shot_evaluate`: prepend a 4-shot prompt (3 TRUE + 1 FALSE, calibrated for Llama-2-13B on `sp_en_trans`), append `"This statement is:"`, read `P(TRUE) - P(FALSE)` from the last-token logits.
- Implemented `intervention_experiment` with a **batch-aware forward hook**: for each batch element, find the right token positions (the statement tokens - *not* padding, *not* few-shot prefix) using `attention_mask`, and add `scale * direction` to the hidden state at those positions at the configured `intervene_layers`.
  - Three modes: `none` (baseline), `add` (push toward TRUE), `subtract` (push toward FALSE).
- Ran the experiment for both the **MM direction** and the **LR direction** of the combined-data probe, scaling both to the same magnitude (project the class-mean difference onto the unit direction).
- Computed **Natural Indirect Effect (NIE)** for each: `NIE_add_on_false = mean P_diff(add, false) - mean P_diff(none, false)` and `NIE_subtract_on_true = mean P_diff(subtract, true) - mean P_diff(none, true)`.

### Results
- Adding the truth direction to *false-statement* activations pushes `P(TRUE) - P(FALSE)` up. Subtracting from *true-statement* activations pushes it down. Both expected - the direction is causal.
- **MM direction has a larger NIE than LR**, despite LR scoring at least as well on classification. Replicates the paper's headline: *"Mass-mean probe directions are highly causal, with MM outperforming LR and CCS in 7/8 experimental conditions."*

### Why MM beats LR causally (mental model)
- LR optimizes classification accuracy. It can lean on *any* feature that correlates with the label - including features the model itself doesn't actually use downstream.
- MM is a coarse "where does the population move on average?" direction. It's noisier as a classifier but is forced to be aligned with the *bulk* of the activation shift between classes - which is more likely to be the thing the model is actually computing with.
- This is the most important lesson of the whole section: **a more accurate probe is not a better probe.** "Better" depends on whether you want correlational or causal evidence.

### Learning
- For interventions, get the masking right or nothing makes sense. Padding tokens, few-shot prefix tokens, and statement tokens all live in the same tensor; only the statement tokens should be patched.
- Hook outputs are sometimes a `tuple` (transformer layer) and sometimes a plain `Tensor` - handle both.

---

## 4. Probing for deception

Switch model: unload Llama-2-13B base, load **Llama-3.1-8B-Instruct**.

### What I did
- Used `utils.build_detection_mask` to robustly identify **assistant-content tokens** in a chat-templated conversation: it formats the conversation with and without the assistant message, diffs the strings, and uses `tokenizer.char_to_token` to find the corresponding token span. Avoids fragile token-count heuristics.
- Built `ChatActivations` dataclass to hold `(text, tokens, attention_mask, detection_mask)`.
- Implemented `construct_instructed_pairs`: for each fact, build two chats with identical user + assistant text but different system prompts:
  - **Honest:** `"You are an honest AI assistant."`
  - **Dishonest:** `"You are a deceptive AI assistant."`
  - Assistant content is the fact **minus its last 5 words** ("fact_start"), forcing the model to complete it under whichever persona.
  - Extract hidden states at the configured layer, average over assistant-content tokens (using the detection mask), one vector per (fact, persona).
- Probed at `mid_layer = INSTRUCT_NUM_LAYERS // 2 = 16` (of 32). The paper's pre-trained 70B probe sits at layer 22/80 (~28th percentile) - naively scaling proportionally across model sizes is not exactly right; sweep when you can.
- Trained `MMProbe` and `LRProbe(C=0.001)` on `(dishonest=1, honest=0)`.
- **Bonus**: multi-layer probe - trained a separate LR probe at each layer in the middle 50% (`INSTRUCT_DETECT_LAYERS`) and **averaged the per-layer scores**.
- Implemented `DeceptionSteeringHook` to add `coef * direction` to the residual stream at a layer, with clean enable/disable lifecycle (`register_forward_hook` + `remove`). Steered model generations with negative coefficients (toward honesty) and positive coefficients (toward deception).
- Implemented `score_dialogue`: per-token probe scores for a full chat, plus an assistant-only mask, so we can visualize *where in the response the probe fires*.
- Evaluated on **AI Liar** (`how_to_catch_an_ai_liar/goal_directed_lying.json`): scenarios where the model is given context that pressures it to lie (e.g. selling a product it knows is harmful).

### Results
- Train/test accuracies are very similar across MM and LR on the instructed-pairs data (binary task, well-determined boundary).
- Steering: at moderate coefficients, behavioral changes are coherent - negative coefficients make outputs more cautious / more willing to admit uncertainty; positive coefficients produce more confident assertions and more willingness to make false claims. Too-large coefficients break grammar.
- AI Liar per-token scores: **the probe fires before any explicitly deceptive text is generated** - the deceptive intent is encoded in the activations early in the response, even at tokens that look innocuous in isolation. This is the most striking qualitative finding.
- 70B (pre-trained `detector.pt`, LR + StandardScaler at layer 22) achieves AUROC 0.96–0.999 on the paper's eval datasets. We did not run 70B locally (commented-out cells, requires int8 quantization via bitsandbytes).

### Learning
- The training data is *extremely simple* - identical assistant text, just two different system prompts. Yet the probe transfers to a complex goal-directed-lying eval. This is what makes the result interesting: the probe is picking up a general "intent to deceive" representation, not memorizing surface patterns.
- Use `apply_chat_template` + `char_to_token` for token masking on chat models. Don't try to count special tokens by hand - chat templates differ across model families.
- Always average over the right tokens. For instructed pairs, averaging over assistant content (not the system prompt!) is what isolates the per-fact representation.

---

## 5. Attention probes for high-stakes detection

Different target: not a property of the model's state, but a property of the **input** ("is this user request high-stakes?"). Same activations, but pool them differently.

### What I did
- Dataset: `Arrrlex/models-under-pressure` on HuggingFace. Balanced 1500/500 train/test per class.
- Implemented `extract_full_sequence_activations`: returns `(acts: (n, seq, d_model), mask: (n, seq))` - the full sequence, not just the last token. Important details: pad/truncate to `max_length`, return on CPU as float32, never silently downcast.
- Probed at layer 16 of Llama-3.1-8B-Instruct (midpoint).
- Implemented `attention_probe_forward(x, mask, W_q, W_out, b_out, scale)` - the **AttnLite** architecture:
  1. **Attention logits**: project each token through `W_q ∈ R^(d_model × n_heads)` → `(batch, seq, n_heads)`.
  2. **Mask + softmax** over the sequence dim (set logits on padded positions to `-inf` *before* softmax - masking afterward leaks probability mass).
  3. **Weighted sum**: per-head, weighted sum of token activations → `(batch, n_heads, d_model)`. Flatten to `(batch, n_heads * d_model)`.
  4. **Output**: linear → scalar logit. `scale` is a learnable temperature on attention logits (start small, e.g. 1/sqrt(d_model)).
- Wrapped it in an `AttentionProbe(nn.Module)` and trained with **AdamW + BCEWithLogitsLoss** for 200 epochs on the pre-extracted tensors (no further LLM forward passes - fast).
- Baselines: **last-token LR**, **mean-pool LR** (mean over real tokens), and **last-token MM**. All trained on the same hidden states.
- Compared by **AUROC** on the test set, then **OOD AUROC** on `anthropic_hh_balanced` and `toolace_balanced` from the same HF dataset (no retraining).
- Inspected learned attention weights on correctly-classified high-stakes prompts.

### Results
- On the held-out synthetic test set, the ordering is approximately: **AttnLite ≥ mean-pool ≈ last-token (LR) > last-token (MM)** by AUROC. Attention adds a small but consistent edge over mean-pool, with much more headroom on the longer / structurally varied prompts.
- OOD: gap between AttnLite and mean-pool narrows, sometimes flips. Real-world distributions are messier than the synthetic training data. Still, mean-pool itself does respectably - suggesting the high-stakes signal is distributed across many tokens, not concentrated in one.
- Attention weights are **mostly uniform-ish**, with two patterns:
  - **Down-weighting very early tokens** (system prompt boilerplate, beginning of user turn) - those positions haven't aggregated semantic context yet.
  - **Up-weighting "signalling words"** ("emergency", "damage", "urgent", numeric stakes, role-of-user clauses) when present.

### Learning
- When in doubt about how to pool, mean-pool is a strong baseline. Last-token is a weak baseline for *input* properties - it's an artifact of the GoT setup that worked well there because declarative statements end in a token that aggregates everything.
- Attention probes are essentially a **learned weighted mean**. The fact that it barely beats unweighted mean means the high-stakes signal is broadly distributed, which is consistent with the near-uniform attention patterns.
- "Probe target" matters a lot. Truth/deception are model-state properties (read off the assistant's representation). High-stakes is an input property (read off the prompt). Same tooling, different inductive bias for pooling.

---

## Cross-cutting takeaways

1. **Truth is linearly represented**, concentrated in early-to-mid layers (roughly 25–50% depth).
2. **Cross-dataset generalization** is the real test of "did we find a universal direction" - improves with scale. In-distribution accuracy is a weak signal.
3. **Classification accuracy ≠ causal relevance.** Always validate causally if you care about mechanism (interventions, steering, ablations).
4. **MM ≥ LR for causality, LR ≥ MM for classification.** LR overfits to any correlation; MM is forced toward the bulk activation shift.
5. **Same toolkit transfers** from factual truth → strategic deception → high-stakes detection, modulo:
   - Target type (model state vs. input property) - affects pooling.
   - Model type (base vs. instruct) - affects token masking and prompt construction.
   - Layer choice - sweep.
6. **The probe fires early.** In deception detection, deceptive intent shows up in activations before any deceptive token is emitted. This is the property that makes probes useful as monitoring tools, not just post-hoc classifiers.
7. **Get the masking right.** Last-real-token, assistant-content-only, statement-tokens-not-padding - the masking decisions silently determine whether you measure the thing you think you're measuring.

---

## Mini-glossary

- **MM probe (mass-mean / difference-of-means):** direction = `mean(class_1) - mean(class_0)`. Classify by sign of dot product.
- **LR probe:** logistic regression on hidden states, usually with `StandardScaler` normalization.
- **CCS:** Contrastive Consistent Search. Unsupervised, paired data, consistency + confidence loss. Caveat: finds prominent features, not necessarily truth.
- **NIE (Natural Indirect Effect):** `mean(score | intervene) - mean(score | no intervene)`. The causal version of "did this direction matter?"
- **AttnLite:** attention probe with a single learned position-independent query vector and a linear classifier on the weighted sum.
- **AUROC:** area under ROC curve. Threshold-free classification metric, good for imbalanced eval.
