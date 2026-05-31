# arena-practice

My working notes and exercise solutions for [ARENA 3.0](https://github.com/callummcdougall/ARENA_3.0). Each directory contains a notebook (the exercises) and a `notes.md` (what I did, key code snippets, results, learnings).

| Dir | ARENA chapter | Notebook | Notes |
|-----|---------------|----------|-------|
| [prerequisites/](prerequisites/) | [0.0] Prerequisites | [0_0_Prerequisites_exercises.ipynb](prerequisites/0_0_Prerequisites_exercises.ipynb) | [notes.md](prerequisites/notes.md) |
| [transformer_from_scratch/](transformer_from_scratch/) | [1.1] Transformers from Scratch | [1_1_Transformer_from_Scratch_exercises.ipynb](transformer_from_scratch/1_1_Transformer_from_Scratch_exercises.ipynb) | [notes.md](transformer_from_scratch/notes.md) |
| [intro_to_mech_interp/](intro_to_mech_interp/) | [1.2] Intro to Mech Interp | [1_2_Intro_to_Mech_Interp_exercises.ipynb](intro_to_mech_interp/1_2_Intro_to_Mech_Interp_exercises.ipynb) | [notes.md](intro_to_mech_interp/notes.md) |
| [linear_probe/](linear_probe/) | [1.3.1] Linear Probes | [1_3_1_Linear_Probes_exercises.ipynb](linear_probe/1_3_1_Linear_Probes_exercises.ipynb) | [notes.md](linear_probe/notes.md) |
| [function_vector_&_model_steering/](function_vector_&_model_steering/) | [1.3.2] Function Vectors & Model Steering | [1_3_2_Function_Vectors_&_Model_Steering_exercises.ipynb](function_vector_&_model_steering/1_3_2_Function_Vectors_&_Model_Steering_exercises.ipynb) | [notes.md](function_vector_&_model_steering/notes.md) |

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

## Recurring lessons across all five

- **Off-by-one is the universal bug.** Logits at position `j` predict token `j+1`. Loss, eval, intervention masking, and induction-stripe offsets all hinge on this.
- **`keepdim=True` and `mask BEFORE softmax`.** Two small habits that prevent a surprising fraction of bugs.
- **Classification accuracy is not causal relevance.** Probes that classify equally well can have very different downstream effects under intervention.
- **Three kinds of evidence converge in mech interp.** Observation (attention patterns), additive decomposition (logit attribution), and causal intervention (ablation). A confident claim wants all three pointing the same way.
- **Get the masking right or measure nothing useful.** Last-real-token, assistant-content-only, statement-tokens-not-padding, induction-stripe-at-offset-`-(seq_len-1)` - the masking choices silently determine what's actually being measured.
- **Pre-norm residual stream is the protagonist** of a transformer. Layers read it (after LayerNorm) and add to it. Everything else is decoration.
- **`FactoredMatrix` makes circuit analysis tractable.** Many interesting matrices are `(d_vocab, d_vocab)` = unmaterializable. Keep them factored and let the library handle norms, SVDs, and indexed slices lazily.
- **The residual stream is a privileged additive bus.** Both function vectors and activation-addition steering work because adding a vector to the residual stream at the right layer composes cleanly with the rest of the forward pass. Gradient-free interventions exploit exactly this.
