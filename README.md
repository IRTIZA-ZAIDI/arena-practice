# arena-practice

My working notes and exercise solutions for [ARENA 3.0](https://github.com/callummcdougall/ARENA_3.0). Each directory contains a notebook (the exercises) and a `notes.md` (what I did, key code snippets, results, learnings).

For an API cheat sheet across all the exercises that use `transformer_lens`, see [transformerlens.md](indirect_object_identification/transformerlens.md).

| Dir | ARENA chapter | Notebook | Notes |
|-----|---------------|----------|-------|
| [prerequisites/](prerequisites/) | [0.0] Prerequisites | [0_0_Prerequisites_exercises.ipynb](prerequisites/0_0_Prerequisites_exercises.ipynb) | [notes.md](prerequisites/notes.md) |
| [transformer_from_scratch/](transformer_from_scratch/) | [1.1] Transformers from Scratch | [1_1_Transformer_from_Scratch_exercises.ipynb](transformer_from_scratch/1_1_Transformer_from_Scratch_exercises.ipynb) | [notes.md](transformer_from_scratch/notes.md) |
| [intro_to_mech_interp/](intro_to_mech_interp/) | [1.2] Intro to Mech Interp | [1_2_Intro_to_Mech_Interp_exercises.ipynb](intro_to_mech_interp/1_2_Intro_to_Mech_Interp_exercises.ipynb) | [notes.md](intro_to_mech_interp/notes.md) |
| [linear_probe/](linear_probe/) | [1.3.1] Linear Probes | [1_3_1_Linear_Probes_exercises.ipynb](linear_probe/1_3_1_Linear_Probes_exercises.ipynb) | [notes.md](linear_probe/notes.md), [linear_probe.excalidraw](linear_probe/linear_probe.excalidraw) |
| [function_vector_&_model_steering/](function_vector_&_model_steering/) | [1.3.2] Function Vectors & Model Steering | [1_3_2_Function_Vectors_&_Model_Steering_exercises.ipynb](function_vector_&_model_steering/1_3_2_Function_Vectors_&_Model_Steering_exercises.ipynb) | [notes.md](function_vector_&_model_steering/notes.md) |
| [interpretability_with_SAEs/](interpretability_with_SAEs/) | [1.3.3] Interpretability with SAEs | [1_3_3_Interpretability_with_SAEs_exercises.ipynb](interpretability_with_SAEs/1_3_3_Interpretability_with_SAEs_exercises.ipynb) | [notes.md](interpretability_with_SAEs/notes.md), [interpretability_with_SAEs.excalidraw](interpretability_with_SAEs/interpretability_with_SAEs.excalidraw) |
| [activation_oracles/](activation_oracles/) | [1.3.4] Activation Oracles | [1_3_4_Activation_Oracles_exercises.ipynb](activation_oracles/1_3_4_Activation_Oracles_exercises.ipynb) | [notes.md](activation_oracles/notes.md), [activation_oracles.excalidraw](activation_oracles/activation_oracles.excalidraw) |
| [indirect_object_identification/](indirect_object_identification/) | [1.4.1] Indirect Object Identification | [1_4_1_Indirect_Object_Identification_exercises.ipynb](indirect_object_identification/1_4_1_Indirect_Object_Identification_exercises.ipynb) | [notes.md](indirect_object_identification/notes.md) |
| [sae_circuits/](sae_circuits/) | [1.4.2] SAE Circuits | [1_4_2_SAE_Circuits_exercises.ipynb](sae_circuits/1_4_2_SAE_Circuits_exercises.ipynb) | [notes.md](sae_circuits/notes.md), [circuit-tracer.md](sae_circuits/circuit-tracer.md), [sae_circuits.excalidraw](sae_circuits/sae_circuits.excalidraw) |
| [balanced_bracket/](balanced_bracket/) | [1.5.1] Balanced Bracket Classifier | [1_5_1_Balanced_Bracket_Classifier_exercises.ipynb](balanced_bracket/1_5_1_Balanced_Bracket_Classifier_exercises.ipynb) | [notes.md](balanced_bracket/notes.md), [balanced_bracket.excalidraw](balanced_bracket/balanced_bracket.excalidraw) |

---

## [prerequisites/](prerequisites/) - tensor fluency

Getting comfortable with the building blocks before anything else.

**Libraries**
- `einops` (rearrange / repeat / reduce) and `einops.einsum`
- `torch` tensor ops (indexing, broadcasting, `gather`, `norm`)
- `numpy` (interop with torch)

**Skills**
- Reading and writing einops patterns. The `(a b)` vs `(b a)` ordering distinction (outer-vs-inner). Factored axes for pooling and tiling.
- Broadcasting rules + `keepdim=True` discipline on reductions.
- Integer indexing (`tensor[idx]`, `matrix[tuple(coords.T)]`) vs `torch.gather` (per-row lookup).
- Numerically stable `logsumexp` / `softmax` / `logsoftmax` / cross-entropy (subtract per-row max before exp).
- No-loop categorical sampling via `cumsum + (rand > cum).sum()`.
- Einsum: the 4 rules and the standard patterns (matmul, batched matmul, attention scores, batched dot).

---

## [transformer_from_scratch/](transformer_from_scratch/) - GPT-2 reimplementation

A from-scratch GPT-2 in PyTorch that's `load_state_dict`-compatible with the TransformerLens reference. Then training a tiny version on TinyStories, then sampling from it.

**Libraries**
- `transformer_lens` (`HookedTransformer`, `tokenize_and_concatenate`)
- `torch.nn` (subclassing `nn.Module`, `nn.Parameter`, buffers)
- `einops` (the Q/K/V projections, attention contractions)
- `transformers` (GPT-2 tokenizer)
- `datasets` (TinyStories)
- `wandb` (training-loop logging)
- `circuitsvis` (attention pattern visualization)

**Skills**
- Implementing every layer: `LayerNorm` (with the `sqrt(var + eps)` gotcha), `Embed` / `PosEmbed`, multi-head causal `Attention` with `apply_causal_mask` (`triu(diagonal=1)`), `MLP` (GELU + 4x expansion), `TransformerBlock` (pre-norm + residual), `Unembed`, `DemoTransformer`.
- Matching parameter conventions exactly so `load_state_dict` works against the reference.
- Q/K/V shape convention `(batch, pos, head, d_head)` so per-head biases broadcast.
- Writing a training loop: AdamW with weight decay, `get_log_probs` cross-entropy with the **off-by-one shift** (`logits[:, :-1]` vs `tokens[:, 1:]`), eval as argmax-accuracy, wandb logging.
- Reading the loss curve as a diagnostic (uniform -> unigram -> bigram knees).
- Implementing all the standard samplers: greedy, basic categorical, temperature, frequency penalty, top-k, top-p (nucleus), beam search with `no_repeat_ngram_size`. KV-caching as a bonus.

---

## [intro_to_mech_interp/](intro_to_mech_interp/) - TransformerLens + induction circuits

The induction-head reverse-engineering walk: discover the circuit by observation, confirm it with hooks and ablations, then read it off the weight matrices directly.

**Libraries**
- `transformer_lens` (`HookedTransformer`, `ActivationCache`, `run_with_hooks`, `utils.get_act_name`, `FactoredMatrix`)
- `circuitsvis` (`cv.attention.attention_patterns` for per-head visualization)
- `einops` + `eindex` (the per-batch-per-position correct-logprob lookup)
- `torch` forward hooks via TransformerLens's `fwd_hooks` API

**Skills**
- Loading and inspecting `HookedTransformer` models, indexing the `ActivationCache` by shorthand (`cache["pattern", 0]`) or full hook name.
- Hand-recomputing attention from cached `q`, `k` to verify the QK math (dot product, scale, causal mask, softmax).
- Building **head-type detectors** (current / previous / first / induction) by averaging attention on the right diagonal (`.diagonal(offset=-k)`).
- Generating **repeated-random-token probes** `[BOS, *rand, *rand]` to test in-context learning, then verifying the loss drop on the second half.
- Writing hook functions that aggregate per-head scores into a global tensor and dispatching them with a name filter (`lambda name: name.endswith("pattern")`).
- **Logit attribution**: decomposing each correct-next-token logit into direct-path + per-head-per-layer contributions.
- **Head ablation** (zero and mean) at the `z` hook to measure each head's causal contribution to loss.
- Reverse-engineering circuits at the weight level: OV copying circuit (`W_E @ W_V @ W_O @ W_U`), QK previous-token circuit (`W_pos @ W_QK @ W_pos.T`), full K-composition circuit, effective OV across multiple heads.
- Using `FactoredMatrix` to handle `(d_vocab, d_vocab)` weight products without materialization.
- Decomposing the QK input to layer 1 into 14 sources (embed + pos_embed + 12 L0 heads), then computing decomposed attention scores `[q_comp, k_comp, q_pos, k_pos]` to identify the dominant pair (signature of K-composition).
- **Composition scores** (`||A B||_F / (||A||_F ||B||_F)`) for all (L0, L1) head pairs in Q/K/V variants, with a Kaiming-init random baseline.
- **Targeted ablations**: ablate one L0 head, measure how the L1 induction head's pattern changes (vs measuring loss).

---

## [linear_probe/](linear_probe/) - mech-interp probing

Reproducing pieces of three papers (Geometry of Truth, Deception Detection, High-Stakes Probes). Training linear classifiers on LLM hidden states, then doing causal interventions and attention-pooled probing.

**Libraries**
- `transformers` (loading Llama-2-13B base, Llama-3.1-8B-Instruct), `bitsandbytes` (int8 for 70B)
- `torch` forward hooks (`register_forward_hook`, careful cleanup in `finally`)
- `sklearn` (`LogisticRegression`, `StandardScaler`)
- `einops` (PCA / attention probe contractions)
- `plotly` (PCA scatter, generalization heatmaps, per-token visualizations)

**Skills**
- Extracting activations cleanly: last-real-token via `attention_mask.sum(-1) - 1`, hidden-state indexing with the `+1` offset (index 0 is embeddings).
- PCA via eigendecomposition of the centered covariance matrix.
- Implementing **MMProbe** (mass-mean / difference-of-means with pooled within-class covariance) and **LRProbe** (logistic regression + StandardScaler, weights baked into a torch module).
- Cross-dataset generalization matrices (train on i, eval on j).
- **Causal interventions**: batch-aware forward hooks that patch only the right token positions (not padding, not few-shot prefix), with the direction scaled by the projected class-mean difference. Hook cleanup in `try/finally`.
- Few-shot eval reading `P(TRUE) - P(FALSE)` from last-token logits.
- Chat-model token masking via `apply_chat_template` + `char_to_token`. Instructed-pairs construction (honest/deceptive system prompts, identical user/assistant text).
- Per-token dialogue scoring with assistant-content masks.
- Steering hooks with per-position activation-norm scaling and clean enable/disable.
- **Attention probes**: extracting full-sequence activations, implementing AttnLite (single learned query, position-independent, mask-before-softmax), training with AdamW + BCE.

---

## [function_vector_&_model_steering/](function_vector_&_model_steering/) - inference-time steering

Two gradient-free intervention paradigms: **function vectors** (Todd et al.) on GPT-J-6B for ICL task transfer, and **activation addition / steering vectors** (Turner et al.) on GPT-2-XL for concept steering.

**Libraries**
- `nnsight` (`LanguageModel`, `model.trace(...)`, `tracer.invoke(...)`, `model.generate(...)`, `.save()` proxy persistence)
- `transformers` (the underlying HuggingFace models: GPT-J-6B, GPT-2-XL)
- `einops` (per-head z reshapes and the CIE stack rearrange)
- `circuitsvis` (attention pattern visualization with the `Ġ` -> space substitution for BPE)

**Skills**
- Reading and writing the `nnsight` trace/invoke pattern. One trace = one batched forward pass with multiple sequential sub-passes that share state; intervention code is interleaved with the model's forward execution.
- Building **ICL datasets** (`ICLSequence`, `ICLDataset`) with clean / corrupted variants. Corrupted = same surface form but shuffled answers in non-final demonstrations.
- Implementing the **h-vector**: mean residual-stream hidden state at the final token across successful ICL prompts. Adding it back to a zero-shot prompt induces task-correct completions.
- Combining "compute h" and "intervene with h" into a single trace via three sequential invokes (so h never leaves the GPU).
- **Causal Indirect Effect (CIE)** for per-head attribution: clean run -> store mean per-head z at final position via `out_proj.input`; corrupted baseline -> measure correct-token logprob; intervened runs -> patch one head's final-position z with the clean mean, measure logprob improvement. All `n_layers * n_heads` interventions inside one trace.
- Building the **function vector** from a list of `(layer, head)` pairs: zero out non-selected heads' z's, push through `out_proj`, sum across layers. Lives in residual-stream space.
- **Multi-token generation interventions** via `model.generate(...)` and `generator.invoke(...)`: the intervention fires at every generation step within the block.
- Implementing **activation-addition steering vectors**: pass two contrasting prompts through the model, take their layer-`L` activations, add the scaled difference to a third prompt's residual stream during generation. Replicated love/hate, weddings, and Eiffel-Rome examples on GPT-2-XL.

---

## [interpretability_with_SAEs/](interpretability_with_SAEs/) - sparse autoencoders

End-to-end SAE interpretability on GPT-2-small, Gemma-2-2B, and TinyStories: load pretrained SAEs from SAELens, replicate Neuronpedia-style latent dashboards, attribute IOI to specific attention-SAE latents via DLA + ablation + attribution patching, do feature steering with GemmaScope, autointerp with held-out scoring, patch scoping for self-explanation, and finally train a small SAE from scratch.

**Libraries**
- `sae_lens` (`SAE`, `HookedSAETransformer`, `ActivationsStore`, `SAETrainingRunner`, `run_with_cache_with_saes`, `with model.saes(...)`)
- `transformer_lens` (the underlying hooked transformer + hook API)
- `sae_vis` (latent dashboards with max activations, histograms, top logits)
- `umap-learn` + `hdbscan` (decoder-vector clustering for feature-splitting analysis)
- OpenAI / Gemini API for autointerp + scoring
- `circuitsvis` (attention visualization)

**Skills**
- Loading pretrained SAEs (`SAE.from_pretrained(release, sae_id)`) and attaching them via `run_with_cache_with_saes` or the `with model.saes(...)` context manager.
- Reading `hook_sae_acts_post` cheaply with `stop_at_layer=sae.cfg.hook_layer + 1` and `names_filter` so dashboards scale to hundreds of latents.
- Toggling `use_error_term` to control whether the model sees `x_hat` alone or `x_hat + (x - x_hat)`.
- Building Neuronpedia-style dashboards from scratch: activation histogram, max-activating examples with ±buffer context, non-overlapping top-k via `get_k_largest_indices(..., no_overlap=True)`, top/bottom unembed logits, autointerp.
- Implementing attention-SAE analysis with **DFA** (`attn_weight * value @ W_enc_latent`) - per-source-token contribution to a destination latent firing.
- The **IOI playbook**: rank latents by max activation -> filter via Direct Logit Attribution (project decoder through `W_O` then `W_U`-diff) -> confirm causally with single-latent ablation at the relevant `seq_pos`.
- **Attribution patching** as a fast ablation proxy via gradient * activation. Forward + backward once, compute all latents' ablation-effect approximations.
- **Feature steering**: hook the SAE site, add `coeff * sae.W_dec[latent_idx]` to all positions during generation. Reproduced on GemmaScope-2B.
- **Feature splitting** detection across SAE widths (768 to 6144+): three approaches (co-firing positions, same-prompt overlap, autointerp keyword search) + UMAP+HDBSCAN visualization of decoder vectors.
- **Autointerp scoring**: split max-activating examples into explain / score halves; LLM generates a description from one half; LLM scores held-out examples against the description.
- **Patch scoping** for self-explanation: replace residual at the `X` position with the normalized + scaled decoder direction in an instruction-tuned model and let it verbalize the latent. KV-cache-aware (only patch on the first forward).
- **Training an SAE from scratch** with `SAETrainingRunner` on TinyStories-1L MLP-out: tuning `l1_coefficient`, l1 warm-up, expansion factor, and reading training curves (L0, dead-latent fraction, reconstruction MSE, CE-with-vs-without-SAE).

---

## [activation_oracles/](activation_oracles/) - open-ended interp via finetuned interpreters

LoRA-finetune another model (the **Activation Oracle**) to accept a target model's residual-stream activations as input tokens and answer arbitrary natural-language questions about them. Replicates the Karvonen et al. (2025) paper: secret-word extraction from "taboo" finetunes (Figure 1), goal extraction from system-prompt personas, emotion tracking, misalignment detection, model-diffing via activation differences on an emergently misaligned Llama-3.1-8B.

**Libraries**
- `transformers` (Qwen3-8B and Llama-3.1-8B-Instruct as both target and oracle base; `apply_chat_template`)
- `peft` (LoRA loading, `model.load_adapter`, `model.set_adapter`, `model.disable_adapter()`)
- PyTorch forward hooks for activation capture + steering

**Skills**
- The full AO pipeline: forward-hook activation capture with `EarlyStopException`, `?`-token placeholder prefix (`Layer: X\n ? ? ? \n<question>`), `find_pattern_in_tokens` to locate them, norm-matched steering hook at the oracle's layer 1 that skips KV-cached single-token forwards.
- `OracleInput` dataclass + `create_oracle_input` that bundles `input_ids`, `layer`, `steering_vectors`, `positions`.
- Adapter management with PEFT: `model.set_adapter("oracle")` / `model.set_adapter("taboo")` / `model.disable_adapter()` to switch between target/oracle/base on one set of weights.
- Three query modes (`full_seq`, `segment`, `single_token`) and the diagnostic uses of each (full-seq for general behavior, segment for "where in the prompt is X encoded", single-token for token-by-token decoding of representation accumulation).
- **Hypothesis tests** for what the oracle is really reading: segment-after-keyword (rules out embedding-only reading), comparison against logit-lens top-k (rules out "just argmax").
- **Taboo / secret extraction**: load taboo LoRA -> assistant-segment activations -> constrained oracle prompt + `forced_model_prefix` for clean single-word extraction.
- **Goal extraction** with no target LoRA, just a system prompt; segment after the system prompt so the oracle reads behavior, not text.
- **Misalignment detection** via the `<|im_end|>` boundary token as a single-token oracle input site (paper finding: punctuation / end-of-turn tokens are empirically strong probe positions).
- **Model-diffing**: collect base activations (adapters disabled) and finetuned activations (EM LoRA active), subtract, feed the difference into the oracle. Describes what the finetune changed.
- Sweep across secret words and layer fractions (10/25/50/75/90%) -> heatmap. Confirms middle layers are best (matches the oracle's training distribution at 25/50/75% depths).

---

## [indirect_object_identification/](indirect_object_identification/) - reverse-engineering the IOI circuit

End-to-end replication of the Wang et al. (2022) IOI circuit on GPT-2-small: logit attribution -> activation patching -> path patching -> head validation -> minimal-circuit construction. The canonical "early-mid-late" picture (duplicate / induction -> S-inhibition -> name movers) emerges concretely.

**Libraries**
- `transformer_lens` (the heavy use - `HookedTransformer`, `ActivationCache` with `accumulated_resid` / `decompose_resid` / `stack_head_results` / `apply_ln_to_stack`, the `patching` module, hand-rolled `run_with_hooks` for path patching, `IOIDataset` from the part34 utils)
- `einops` (per-head rearrangements, logit-direction projections)
- `circuitsvis` (per-head attention pattern visualization)

**Skills**
- Loading GPT-2-small with the four interp-friendly flags (`center_unembed`, `center_writing_weights`, `fold_ln`, `refactor_factored_attn_matrices`).
- Building ABBA/BABA paired prompts + an `answer_tokens` `(IO, S)` tensor; defining the `logits_to_ave_logit_diff` metric and the `(diff - corrupted) / (clean - corrupted)` calibrated `ioi_metric`.
- **Direct logit attribution**: project residual-stream components onto `W_U[:, IO] - W_U[:, S]` *after* `cache.apply_ln_to_stack(..., layer=-1)`. Three views via `accumulated_resid` (logit lens by layer), `decompose_resid` (per-component), `stack_head_results` (per (layer, head) heatmap).
- **Activation patching** sweeps via `patching.get_act_patch_resid_pre`, `get_act_patch_block_every`, `get_act_patch_attn_head_out_all_pos`; hand-rolled versions for `patch_residual_component` and `patch_head_vector` using `model.run_with_hooks`.
- **Path patching**: three-pass algorithm that freezes all attention heads to clean values except one sender, captures the modified `resid_post[-1]`, then swaps it back into a fresh clean run. Identifies Name Mover heads (head -> final residual) and the K-composition feeding S-Inhibition heads (head -> downstream V).
- **Head validation** via writing-direction scatter plots (attention to IO vs IO-direction logit contribution), `W_E @ W_OV @ W_U` copying-score matrices, attention-pattern-shape checks for duplicate-token / induction / S-inhibition / previous-token heads.
- **Minimal circuit construction**: mean-ablate non-circuit heads (using ABC-corrupted mean), measure faithfulness/completeness/minimality.

---

## [sae_circuits/](sae_circuits/) - assembling SAE latents into circuits

Four passes at the "how do latents compose across layers" question: pairwise latent-Jacobians, transcoder pullbacks (input-free), full attribution graphs on a linearized model, and end-to-end circuit-tracing on the Dallas/Austin two-hop factual-recall circuit. Companion tutorial for the library used in Section 4: [circuit-tracer.md](sae_circuits/circuit-tracer.md).

**Libraries**
- `sae_lens` (residual-stream SAEs on GPT-2, MLP transcoders on GPT-2, GemmaScope 2 transcoders on Gemma-3-1B-IT)
- `torch.func.jacrev` with `has_aux=True` for the sparse latent Jacobians
- `transformer_lens` for the underlying hooked model + freeze/replacement hooks
- `circuit_tracer` (`ReplacementModel`, `attribute`, `feature_intervention`, `feature_intervention_generate`)
- `hdbscan` + `umap-learn` (feature-cluster analysis) - unchanged from the earlier SAE exercises

**Skills**
- Cross-layer **latent Jacobians** via a `SparseTensor` representation and `torch.func.jacrev(fn, argnums=0, has_aux=True)`. Applied to latent-to-latent, token-to-latent, and latent-to-logit variants.
- **Transcoders**: input-free pullbacks `sae_lower.W_dec[i] @ sae_upper.W_enc[:, j]`, de-embedding `W_E @ W_enc[:, latent]`, and the **extended embedding** trick (`W_E + MLP_0(W_E)`) to break GPT-2's tied-embedding degeneracy.
- **Blind case study**: interpret a latent using four converging analyses (de-embedding, logit lens, pullback, attention attribution) with no other information.
- **Linearizing the model**: `FreezeHooks` (attention patterns + LayerNorm scales cached and replayed) + `TranscoderReplacementHooks` (MLP forward stays exact via skip-connection trick, backward routed through the linear surrogate). Preserves forward-pass logits within 1e-4.
- Building an **attribution graph**: reading/writing vector abstraction for 4 node types (embedding, latent, MLP error, logit), demeaned unembed columns for output nodes, batched backward passes to compute the adjacency matrix, row-normalize + Neumann-series influence + two-stage pruning.
- **`circuit-tracer` library**: `ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="transformerlens")`, `attribute(prompt, model)`, `feature_intervention(prompt, [(layer, pos, feat_idx, value)...])`, `feature_intervention_generate` with sliced positions for sustained interventions.
- **Supernode-level ablations** on the Dallas/Austin circuit: predict-then-test protocol; cross-prompt feature swaps (Texas -> California -> Sacramento, Texas -> China -> Beijing) demonstrating compositional slot-filling.
- **Manual attribution matrices** (bonus): explicit component-by-component propagation (`map_through_ln`, `map_through_attn`, `map_through_mlp`) as a verification path for the gradient-based method.

---

## [balanced_bracket/](balanced_bracket/) - toy-model reverse engineering

Reverse-engineering a 3-layer bidirectional transformer that classifies whether a bracket string is balanced. Discovers the "elevation circuit": head 0.0 tallies opens minus closes over the suffix, MLPs 0/1 threshold that signal nonlinearly, head 2.0 copies the result from position 1 to position 0 where the classifier reads it. First BERT-style (bidirectional, [CLS]-at-position-0) target model in the curriculum.

**Libraries**
- `transformer_lens` (`HookedTransformer` with a custom pretrained bracket classifier, hooked activations, `utils.get_act_name`, `ActivationCache`)
- `sklearn.linear_model.LinearRegression` (linear approximation of LayerNorm for direction analysis)
- `einops` (per-component decompositions of the residual stream)
- `plotly` (histograms per component, attention patterns, neuron activation curves)
- `torch` forward hooks for activation patching on Q vs K

**Skills**
- Reading a **bidirectional** transformer: no causal mask, but padding IS masked. Classification at seq position 0 only (like BERT's [CLS]).
- **Back-propagating a direction** through the classifier tail: `post_final_ln_dir = W_U[:, 0] - W_U[:, 1]`, then through a **linear fit to LayerNorm** to get `pre_final_ln_dir`, then through `W_OV` of head 2.0 + LN2 fit to get `pre_20_dir` at position 1.
- **Linear approximation of LayerNorm**: fit `sklearn.LinearRegression` from LN input to LN output; verify R^2 approx 1; use the coefficient matrix as `L_final` so LN acts like a linear map for direction analysis.
- **Component decomposition of the residual stream** into 10 additive terms (1 embed + 6 heads + 3 MLPs); per-component histograms of dot product with the unbalanced direction to find responsible components.
- **Failure-mode classification** for unbalanced strings: `total_elevation_failure` (unequal counts) vs `negative_failure` (unmatched open bracket reading right-to-left). Split histograms by failure type to identify which components handle which failure mode.
- **OV-circuit analysis**: `W_OV = W_V @ W_O`; use `emb(char) @ L0.T @ W_OV` to see which residual direction each character token gets mapped to. Cosine similarity of `v_L`, `v_R` approx -1 reveals head 0.0's OV is one-dimensional (a signed elevation).
- **Per-neuron analysis** in MLPs: `f(x @ W_in + b_in) * W_out[i, :]` per neuron; project onto the target direction; plot per-neuron output against the sequence's open-proportion to find "threshold detector" neurons.
- **Attention-pattern reading** for head 2.0 (attends only to position 1 from query 0 = "just moves info") and head 0.0 (uniform attention over the suffix = "tally averaged over positions").
- **Activation patching on Q vs K** to figure out which side of an attention head drives a given behavior.
- Working out the full circuit forwards: embed -> head 0.0 uniform-attention writes elevation to resid[1] -> MLPs 0/1 threshold nonlinearly -> head 2.0 moves signal to resid[0] -> classifier reads unbalanced direction.

---

## Recurring lessons across all ten

- **Off-by-one is the universal bug.** Logits at position `j` predict token `j+1`. Loss, eval, intervention masking, and induction-stripe offsets all hinge on this.
- **`keepdim=True` and `mask BEFORE softmax`.** Two small habits that prevent a surprising fraction of bugs.
- **Classification accuracy is not causal relevance.** Probes that classify equally well can have very different downstream effects under intervention.
- **Three kinds of evidence converge in mech interp.** Observation (attention patterns), additive decomposition (logit attribution), and causal intervention (ablation). A confident claim wants all three pointing the same way.
- **Get the masking right or measure nothing useful.** Last-real-token, assistant-content-only, statement-tokens-not-padding, induction-stripe-at-offset-`-(seq_len-1)` - the masking choices silently determine what's actually being measured.
- **Pre-norm residual stream is the protagonist** of a transformer. Layers read it (after LayerNorm) and add to it. Everything else is decoration.
- **`FactoredMatrix` makes circuit analysis tractable.** Many interesting matrices are `(d_vocab, d_vocab)` = unmaterializable. Keep them factored and let the library handle norms, SVDs, and indexed slices lazily.
- **The residual stream is a privileged additive bus.** Both function vectors and activation-addition steering work because adding a vector to the residual stream at the right layer composes cleanly with the rest of the forward pass. Gradient-free interventions exploit exactly this.
- **Activation alone is not interpretation.** A probe that classifies, a latent that fires, or a head whose attention pattern matches a target - none of these are causal claims. Always validate with intervention (ablation, attribution patching, steering) before concluding mechanism.
- **Interp tools are only valid in their training distribution.** Probes work at the layers they were trained on; SAEs decode the site they were trained on; Activation Oracles read activations from the layer depths they were trained on (25/50/75% for the AO paper). Going outside the trained range degrades silently rather than loudly.
- **Build claims out of converging evidence, not single experiments.** A head is "a Name Mover" only when logit attribution, path patching, attention-pattern check, and OV-copying scores all agree. Any one alone can mislead; the conjunction is the standard.
- **Predict-then-test on interventions.** Writing down predicted ablation effects before running the code separates real understanding from post-hoc rationalization. Matches the pattern in IOI, activation oracles, and the Dallas/Austin circuit in SAE circuits.
