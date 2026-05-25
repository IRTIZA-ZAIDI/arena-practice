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

### Tasks
- Implemented `extract_activations(statements, model, tokenizer, layers)`:
  - Tokenize with right-padding, run forward pass with `output_hidden_states=True`.
  - For each statement, pull the hidden state at the **last real token** (use `attention_mask.sum(-1) - 1` per row, not just `[:, -1]` - padding lives on the right).
  - Returns `dict[layer_idx -> (n, d_model)]`.
- Probed at `PROBE_LAYER = 14` (the value used in the Geometry-of-Truth config for Llama-2-13B).
- Datasets: `cities`, `sp_en_trans`, `larger_than` - short declarative true/false statements.
- Implemented `get_pca_components` via eigendecomposition of the centered covariance matrix (top-k eigenvectors, sorted by eigenvalue desc).
- Projected activations onto top-2 PCs and plotted, colored by label.
- Implemented `layer_sweep_accuracy`: at every layer, train a **difference-of-means classifier** (direction = `mean(true) - mean(false)`, decide by sign of dot product), report train/test accuracy.

```python
def extract_activations(statements, model, tokenizer, layers, batch_size=25):
    """Last-token hidden states at the requested layers."""
    all_acts = {layer: [] for layer in layers}
    for i in range(0, len(statements), batch_size):
        batch = statements[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(model.device)
        with t.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # attention_mask.sum(dim=1) - 1 = index of last real token (right padding).
        last_token_idx = inputs["attention_mask"].sum(dim=1) - 1
        for layer in layers:
            hidden = outputs.hidden_states[layer + 1]  # +1: index 0 is embeddings
            batch_indices = t.arange(hidden.shape[0], device=hidden.device)
            acts = hidden[batch_indices, last_token_idx]
            all_acts[layer].append(acts.cpu().float())
    return {layer: t.cat(acts_list, dim=0) for layer, acts_list in all_acts.items()}

def get_pca_components(activations, k=2):
    """Top-k principal components via covariance eigendecomposition."""
    X = activations - activations.mean(dim=0)               # mean-center
    cov = X.t() @ X / (X.shape[0] - 1)                      # (d, d)
    eigenvalues, eigenvectors = t.linalg.eigh(cov)
    sorted_indices = t.argsort(eigenvalues, descending=True)
    return eigenvectors[:, sorted_indices[:k]]              # (d, k)

def layer_sweep_accuracy(statements, labels, model, tokenizer, layers, train_frac=0.8):
    """Train a DoM classifier at each layer, report train/test accuracy."""
    n_train = int(len(statements) * train_frac)
    perm = t.randperm(len(statements))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    train_stmts = [statements[i] for i in train_idx]
    test_stmts = [statements[i] for i in test_idx]
    train_y, test_y = labels[train_idx], labels[test_idx]

    train_acts = extract_activations(train_stmts, model, tokenizer, layers)
    test_acts = extract_activations(test_stmts, model, tokenizer, layers)

    out = {"train_acc": [], "test_acc": []}
    for layer in layers:
        tr, te = train_acts[layer], test_acts[layer]
        true_mean = tr[train_y == 1].mean(dim=0)
        false_mean = tr[train_y == 0].mean(dim=0)
        direction = true_mean - false_mean
        midpoint = (true_mean + false_mean) / 2
        # Classify by sign of (x - midpoint) . direction.
        train_pred = ((tr - midpoint) @ direction > 0).float()
        test_pred = ((te - midpoint) @ direction > 0).float()
        out["train_acc"].append((train_pred == train_y).float().mean().item())
        out["test_acc"].append((test_pred == test_y).float().mean().item())
    return out
```

### Results
- **PCA plots are visibly separable by true/false** for all three datasets, even though PCA is unsupervised. Truth is one of the *most prominent* directions of variance in mid-layer activations.
- **Layer sweep** peaks roughly in layers 8–20 for Llama-2-13B. Layer 14 is near the peak. Very early layers (0–5) and very late layers (35+) probe poorly - early layers haven't computed truth-relevant features yet; late layers have specialized for next-token prediction.

### Learning
- The last-real-token activation is what carries the model's "judgment" of a declarative statement. The end-of-sentence period token in particular aggregates a lot.
- Truth lives **mid-network**, not at the output. This generalizes - for most semantic probes, sweep the layers.
- PCA separation is the cleanest evidence that something is *linearly* represented and *prominent*; if the supervised probe works but PCA doesn't separate, the feature is linear but minor.

---

## 2. Training & comparing probes

### Tasks
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

```python
class MMProbe(t.nn.Module):
    def __init__(self, direction, covariance=None, atol=1e-3):
        super().__init__()
        self.direction = t.nn.Parameter(direction, requires_grad=False)
        self.inv = (t.nn.Parameter(
            t.linalg.pinv(covariance, hermitian=True, atol=atol), requires_grad=False)
            if covariance is not None else None)

    def forward(self, x, iid=False):
        # iid=True applies the within-class covariance correction (whitening).
        if iid and self.inv is not None:
            return t.sigmoid(x @ self.inv @ self.direction)
        return t.sigmoid(x @ self.direction)

    def pred(self, x, iid=False):
        return self(x, iid=iid).round()

    @staticmethod
    def from_data(acts, labels, device="cpu"):
        acts, labels = acts.to(device), labels.to(device)
        pos, neg = acts[labels == 1], acts[labels == 0]
        pos_mean, neg_mean = pos.mean(0), neg.mean(0)
        direction = pos_mean - neg_mean
        # Pooled within-class covariance for IID correction.
        centered = t.cat([pos - pos_mean, neg - neg_mean], dim=0)
        covariance = centered.t() @ centered / acts.shape[0]
        return MMProbe(direction, covariance=covariance).to(device)


class LRProbe(t.nn.Module):
    def __init__(self, d_in, scaler_mean=None, scaler_scale=None):
        super().__init__()
        self.net = t.nn.Sequential(t.nn.Linear(d_in, 1, bias=False), t.nn.Sigmoid())
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)

    def _normalize(self, x):
        if self.scaler_mean is not None and self.scaler_scale is not None:
            return (x - self.scaler_mean) / self.scaler_scale
        return x

    def forward(self, x):
        return self.net(self._normalize(x)).squeeze(-1)

    def pred(self, x):
        return self(x).round()

    @property
    def direction(self):
        return self.net[0].weight.data[0]

    @staticmethod
    def from_data(acts, labels, C=0.1, device="cpu"):
        X, y = acts.cpu().float().numpy(), labels.cpu().float().numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(C=C, random_state=42, fit_intercept=False, max_iter=1000)
        lr.fit(X_scaled, y)
        probe = LRProbe(
            acts.shape[-1],
            scaler_mean=t.tensor(scaler.mean_, dtype=t.float32),
            scaler_scale=t.tensor(scaler.scale_, dtype=t.float32),
        ).to(device)
        probe.net[0].weight.data[0] = t.tensor(lr.coef_[0], dtype=t.float32).to(device)
        return probe


def compute_generalization_matrix(train_acts, train_labels, test_acts, test_labels,
                                  dataset_names, probe_cls):
    n = len(dataset_names)
    matrix = t.zeros(n, n)
    for i, train_name in enumerate(dataset_names):
        probe = probe_cls.from_data(train_acts[train_name], train_labels[train_name])
        for j, test_name in enumerate(dataset_names):
            preds = probe.pred(test_acts[test_name])
            matrix[i, j] = (preds == test_labels[test_name]).float().mean().item()
    return matrix
```

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

### Tasks
- Implemented `few_shot_evaluate`: prepend a 4-shot prompt (3 TRUE + 1 FALSE, calibrated for Llama-2-13B on `sp_en_trans`), append `"This statement is:"`, read `P(TRUE) - P(FALSE)` from the last-token logits.
- Implemented `intervention_experiment` with a **batch-aware forward hook**: for each batch element, find the right token positions (the statement tokens - *not* padding, *not* few-shot prefix) using `attention_mask`, and add `scale * direction` to the hidden state at those positions at the configured `intervene_layers`.
  - Three modes: `none` (baseline), `add` (push toward TRUE), `subtract` (push toward FALSE).
- Ran the experiment for both the **MM direction** and the **LR direction** of the combined-data probe, scaling both to the same magnitude (project the class-mean difference onto the unit direction).
- Computed **Natural Indirect Effect (NIE)** for each: `NIE_add_on_false = mean P_diff(add, false) - mean P_diff(none, false)` and `NIE_subtract_on_true = mean P_diff(subtract, true) - mean P_diff(none, true)`.

```python
def few_shot_evaluate(statements, model, tokenizer, few_shot_prompt,
                      true_id, false_id, batch_size=32):
    p_diffs = []
    for i in range(0, len(statements), batch_size):
        batch = statements[i : i + batch_size]
        prompts = [few_shot_prompt + s + " This statement is:" for s in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(model.device)
        with t.no_grad():
            outputs = model(**inputs)
        last_idx = inputs["attention_mask"].sum(dim=1) - 1
        b_idx = t.arange(len(batch), device=outputs.logits.device)
        probs = outputs.logits[b_idx, last_idx].softmax(dim=-1)
        p_diffs.append((probs[:, true_id] - probs[:, false_id]).cpu().float())
    return t.cat(p_diffs)


def make_batch_hook(dir_vec, attn_mask, scl, len_suffix):
    """Patch only the statement's last 2 tokens (period + first suffix token).
       Per batch element, no padding, no few-shot prefix."""
    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        seq_lens = attn_mask.sum(dim=1)            # real length per batch element
        for b in range(hidden_states.shape[0]):
            end = seq_lens[b].item()               # one past the last real token
            for offset in [-len_suffix, -len_suffix - 1]:
                pos = int(end + offset)
                if 0 <= pos < hidden_states.shape[1]:
                    hidden_states[b, pos, :] += scl * dir_vec
        return (hidden_states,) + output[1:] if isinstance(output, tuple) else hidden_states
    return hook_fn


# Scale a unit direction by the class-mean projection so MM/LR are comparable.
direction_hat = direction / direction.norm()
projection_diff = ((true_mean - false_mean) @ direction_hat).item()
scaled_direction = projection_diff * direction_hat

# Always remove hooks in finally so failures don't leave the model patched.
try:
    hook = model.model.layers[layer_idx].register_forward_hook(
        make_batch_hook(scaled_direction, inputs["attention_mask"], scl=+1.0, len_suffix=5))
    with t.no_grad():
        outputs = model(**inputs)
finally:
    hook.remove()
```

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

### Tasks
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

```python
def construct_instructed_pairs(facts, model, tokenizer, layers):
    honest_acts = {layer: [] for layer in layers}
    dishonest_acts = {layer: [] for layer in layers}
    honest_sys = "You are an honest AI assistant."
    dishonest_sys = "You are a deceptive AI assistant."
    user_msg = "Please tell me a fact."

    for fact in tqdm(facts):
        # Drop the last 5 words: the probe must not see how the fact resolves.
        words = fact.split(" ")
        fact_start = " ".join(words[:-5]) if len(words) > 5 else fact
        for sys_prompt, acts_dict in [(honest_sys, honest_acts),
                                      (dishonest_sys, dishonest_acts)]:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": fact_start},
            ]
            # ChatActivations uses char_to_token to mask assistant-content tokens.
            chat_acts = ChatActivations.from_messages(messages, tokenizer)
            layer_acts = chat_acts.extract_activations(model, layers, average=True)
            for layer in layers:
                acts_dict[layer].append(layer_acts[layer])
    return ({l: t.stack(v) for l, v in honest_acts.items()},
            {l: t.stack(v) for l, v in dishonest_acts.items()})


class DeceptionSteeringHook:
    """Adds a steering vector to the residual stream during forward pass."""

    def __init__(self, steering_vector, layer, steering_coef, apply_to_all_tokens=True):
        self.steering_vector = steering_vector
        self.layer = layer
        self.steering_coef = steering_coef
        self.apply_to_all_tokens = apply_to_all_tokens
        self.hook = None

    def _hook_fn(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        v = self.steering_vector.to(hidden_states.device, dtype=hidden_states.dtype)
        v_normed = v / (v.norm() + 1e-8)
        if self.apply_to_all_tokens:
            # Scale steering by per-position activation norm so it stays proportional.
            norm = t.norm(hidden_states, dim=-1, keepdim=True)
            hidden_states = hidden_states + self.steering_coef * norm * v_normed
        else:
            norm = t.norm(hidden_states[:, -1, :], dim=-1, keepdim=True)
            hidden_states[:, -1, :] = (hidden_states[:, -1, :]
                                      + self.steering_coef * norm * v_normed)
        return (hidden_states,) + output[1:] if isinstance(output, tuple) else hidden_states

    def enable(self, model):
        self.hook = model.model.layers[self.layer].register_forward_hook(self._hook_fn)

    def disable(self):
        if self.hook:
            self.hook.remove()
            self.hook = None


def score_dialogue(messages, model, tokenizer, probe_direction, layer):
    """Per-token probe scores plus an assistant-content mask."""
    prompts = tokenizer.apply_chat_template(messages, tokenize=False)
    assistant_mask = utils.get_assistant_token_mask(messages, tokenizer)
    str_tokens, per_token_scores = utils.score_tokens_with_probe(
        prompts, model, tokenizer, probe_direction, layer)
    return str_tokens, per_token_scores, assistant_mask
```

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

### Tasks
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

```python
def extract_full_sequence_activations(texts, model, tokenizer, layer,
                                      batch_size=1, max_length=128):
    """Returns (acts: (n, max_length, d_model), mask: (n, max_length))."""
    all_acts, all_masks = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=max_length).to(model.device)
        with t.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        all_acts.append(outputs.hidden_states[layer + 1].cpu().float())  # (b, s, d)
        all_masks.append(inputs["attention_mask"].bool().cpu())          # (b, s)
        del outputs, inputs
        t.cuda.empty_cache()
    return t.cat(all_acts, dim=0), t.cat(all_masks, dim=0)


def attention_probe_forward(x, mask, W_q, W_out, b_out, scale):
    """x: (b, s, d), mask: (b, s) bool, W_q: (d, h), W_out: (h*d, 1)."""
    # 1. Per-token, per-head attention logit.
    attn_logits = (x @ W_q) / scale                              # (b, s, h)
    # 2. Mask padding BEFORE softmax (masking after leaks probability mass).
    attn_logits = attn_logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    attn_weights = attn_logits.softmax(dim=1)                    # (b, s, h)
    # 3. Weighted sum over tokens per head, then concat heads.
    context = attn_weights.transpose(1, 2) @ x                   # (b, h, d)
    context = context.flatten(start_dim=1)                       # (b, h*d)
    # 4. Linear classifier.
    logits = (context @ W_out).squeeze(-1) + b_out.squeeze()     # (b,)
    return logits, attn_weights


class AttentionProbe(t.nn.Module):
    def __init__(self, d_model, n_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.scale = d_model ** 0.5
        self.W_q = t.nn.Parameter(t.empty(d_model, n_heads))
        self.W_out = t.nn.Parameter(t.empty(n_heads * d_model, 1))
        self.b_out = t.nn.Parameter(t.zeros(1))
        t.nn.init.normal_(self.W_q, std=d_model ** -0.5)
        t.nn.init.normal_(self.W_out, std=(n_heads * d_model) ** -0.5)

    def forward(self, x, mask):
        logits, _ = attention_probe_forward(x, mask, self.W_q, self.W_out,
                                            self.b_out, self.scale)
        return logits

    @t.no_grad()
    def get_attention_weights(self, x, mask):
        _, attn = attention_probe_forward(x, mask, self.W_q, self.W_out,
                                          self.b_out, self.scale)
        return attn


def train_attention_probe(acts, masks, labels, n_heads=1, n_epochs=200,
                          lr=5e-3, weight_decay=1e-3):
    probe = AttentionProbe(d_model=acts.shape[-1], n_heads=n_heads)
    optim = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = t.nn.BCEWithLogitsLoss()
    probe.train()
    for _ in range(n_epochs):
        logits = probe(acts, masks)
        loss = criterion(logits, labels)
        optim.zero_grad(); loss.backward(); optim.step()
    return probe
```

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
