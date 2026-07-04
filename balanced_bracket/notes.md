# Balanced Bracket Classifier - Notes

Notes from ARENA [1.5.1] Balanced Bracket Classifier. Companion to [1_5_1_Balanced_Bracket_Classifier_exercises.ipynb](1_5_1_Balanced_Bracket_Classifier_exercises.ipynb).

The central question: a tiny 3-layer bidirectional transformer classifies whether a bracket string like `(()())` is balanced. **What algorithm did it learn, and which heads/MLPs implement it?** The exercises reverse-engineer the "elevation circuit" from scratch: back-propagate the unbalanced direction through the classifier -> decompose the residual stream by component -> narrow down to head 2.0 -> narrow further to head 0.0 -> tie the whole story together.

Model: a custom 3-layer bidirectional transformer (BERT-style attention, no causal mask) trained end-to-end on balanced-bracket classification. Loaded via a `HookedTransformer` wrapper so we can hook activations.

For a TransformerLens API cheat sheet referenced throughout these notes, see [../indirect_object_identification/transformerlens.md](../indirect_object_identification/transformerlens.md).

---

## Big picture (the elevation circuit)

The model learned this algorithm:

1. For each position `i`, compute the **elevation of the suffix `seq[i:]`** = (count of `)`) - (count of `(`) in that suffix. Encode this as one number per position.
2. Sequence is unbalanced iff `elevation[0] != 0` (total-elevation failure) OR `any(elevation < 0)` (negative failure).
3. Classify at position 0.

The interp story maps this onto the network:

- **Head 0.0** (early, position 1 as query): with query = `[start]` neighbour, attends uniformly over every position `>= 1`. Its OV circuit copies `-1` for `(` and `+1` for `)`, so the value written into residual-stream position 1 is proportional to (closes minus opens) averaged over the whole suffix - i.e. the **total elevation** signal.
- **MLPs 0 and 1** (position 1): apply a **nonlinear function** of that elevation. Individual neurons fire strongly when the open-proportion is `< 0.5` (or `> 0.5`), effectively thresholding the elevation into "is this 0?" vs "is this non-zero?".
- **Head 2.0** (position 0 as query, attends only to position 1): copies the MLP-processed elevation signal from residual-stream position 1 to position 0 via its W_OV circuit. This is where the "read the answer at position 0" gets set up.
- **Final LayerNorm + W_U + softmax**: the classification head reads that signal at position 0 and outputs P(unbalanced).

Other failure modes (like `())(`) are handled by a different sub-circuit that we don't reverse-engineer in this notebook (touched in the bonus).

The interesting findings:
- The model **works from right to left**, computing elevations over suffixes (not prefixes). The model has no idea what "English reading order" is, so it picked whichever equally-valid direction its gradient nudged it toward.
- **Position 0 is the classification position, position 1 is where the "answer" lives** before head 2.0 moves it. Everything upstream of head 2.0 writes to position 1.
- **Head 2.0's attention pattern is almost degenerate**: it just moves resid[1] -> resid[0]. The interesting work is what wrote to resid[1].
- **Head 0.0's OV circuit is one-dimensional**: `v_L = emb("(") @ W_OV[0,0]` and `v_R = emb(")") @ W_OV[0,0]` are approximately antiparallel. So the head is just tallying `+1` per `)` minus `+1` per `(`, averaged by the uniform attention.

---

## 1. Model architecture (the "toy model")

### Tasks
- Loaded the pretrained bracket classifier (weights + hyperparameters supplied by ARENA).
- Confirmed the architectural details: `d_model=56`, `n_layers=3`, `n_heads=2` per layer, `d_head=28`, `d_mlp=56`, ReLU activation, **bidirectional** attention (no causal mask), **sinusoidal** (non-learned) positional embeddings.
- Vocab is 5 tokens only: `[start]`, `[pad]`, `[end]`, `(`, `)`. Embedding matrix `W_E: (5, 56)`.
- Unembed `W_U: (56, 2)` - two classes: `unbalanced` (index 0) and `balanced` (index 1).
- Classification uses **only position 0** of the output. `logits.shape = (batch, seq_len, 2)`, but we take `logits[:, 0, :]`.

### The key differences from GPT
- **Bidirectional attention**: position `t` can attend to positions before AND after it. No causal mask. This is why we call it "BERT-style" attention.
- **Padding matters**: sequences are padded to the same length. We must mask attention scores at padding positions so the model can't cheat by attending to `[pad]` tokens.
- **Classification, not next-token prediction**: only position 0's output is used. The other positions are computed (they must exist so information can flow bidirectionally) but their outputs are discarded.

```python
# Vocab (5 tokens):
# 0: [start],  1: [pad],  2: [end],  3: (,  4: )
model = load_bracket_classifier()   # HookedTransformer wrapper

# The classification is at seq position 0 only.
logits = model(toks)                # (batch, seq_len, 2)
cls_logits = logits[:, 0, :]        # (batch, 2)   <-- what we care about
```

### Implementing the padding mask
The attention scores get set to `-inf` wherever the KEY is a `[pad]` token, so softmax gives them zero weight. Bidirectional attention doesn't otherwise have any mask - queries at position `t` see keys at every non-pad position.

```python
def make_bidirectional_pad_mask(toks, pad_id=1):
    """
    Return a mask of shape (batch, 1, 1, seq_len) that's -inf at pad key positions.
    Broadcasts over (batch, heads, seq_q, seq_k) in the attention scores.
    """
    is_pad = (toks == pad_id)          # (batch, seq_len)
    mask   = is_pad[:, None, None, :]  # (batch, 1, 1, seq_len)   -> broadcasts as key mask
    return mask
```

---

## 2. Moving backwards: unbalanced direction through the classifier

The classifier is `logits = final_LN(x) @ W_U`, then softmax. To ask "what pushes the model toward saying unbalanced", I back-propagate that question through each stage.

### Stage 1: softmax
`P(unbalanced) = softmax(logits)[0] = sigmoid(logit_0 - logit_1)`. Sigmoid is monotonic, so I only care about the **logit difference** `logit_0 - logit_1`.

### Stage 2: linear (W_U)
Writing out the matrix product:
```
logit_0 - logit_1
  = final_LN_output[0, :] @ W_U[:, 0]  -  final_LN_output[0, :] @ W_U[:, 1]
  = final_LN_output[0, :] @ (W_U[:, 0] - W_U[:, 1])
```

So the **unbalanced direction after the final LN** is:
```python
def get_post_final_ln_dir(model):
    return model.W_U[:, 0] - model.W_U[:, 1]   # shape (d_model,)
```

The bigger the dot product of `final_LN_output[0]` with this direction, the more the model votes "unbalanced".

### Stage 3: LayerNorm (linear approximation)
LayerNorm is `y = ((x - mean) / std) * gamma + beta`. Nonlinear (because of the divide-by-std). But if we ignore the batch-dependent std variation and **fit a linear regression** from the LN input to the LN output, the fit is nearly perfect for our small model (R^2 > 0.99). We can therefore treat LN as approximately `L_final @ x + c` for some matrix `L_final`.

Then the **unbalanced direction BEFORE the final LN** is:
```python
def get_pre_final_ln_dir(model, data):
    L_final, r2 = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)
    L_final = t.from_numpy(L_final.coef_).to(device)
    post_ln_dir = get_post_final_ln_dir(model)
    return L_final.T @ post_ln_dir   # shape (d_model,)
```

Fitting the LN just for `seq_pos=0` is important because the classification uses only position 0.

### Why this back-propagation matters
Once I have `pre_final_ln_dir`, I can ask: for each component that writes to the residual stream (embeddings, each head's output, each MLP's output), **how much does it write in that direction**? Higher dot product = "more responsible for classifying unbalanced".

---

## 3. Writing the residual stream as a sum of components

The residual stream just before the final LN is a sum of 10 things:

```
resid_pre_final =
    (token_embed + pos_embed)          # 1 term (embedding path)
  + layer0.head0                       # 6 heads across 3 layers
  + layer0.head1
  + layer0.mlp                         # 3 MLPs
  + layer1.head0
  + layer1.head1
  + layer1.mlp
  + layer2.head0
  + layer2.head1
  + layer2.mlp
```

That's 10 components (1 embed + 6 heads + 3 MLPs). Each writes an additive contribution to the residual stream. By linearity of dot-product, the total dot product with `pre_final_ln_dir` decomposes as a sum of 10 per-component contributions.

### Getting per-component outputs

```python
def get_out_by_components(model, data):
    """
    Returns (10, batch, seq_pos, d_model): for each of the 10 components,
    the vector it writes to the residual stream.
    """
    names = ["hook_embed", "hook_pos_embed"]
    for layer in range(3):
        names.append(utils.get_act_name("result", layer))   # attn heads
        names.append(utils.get_act_name("mlp_out", layer))  # mlp

    cache = get_activations(model, data.toks, names)

    # Sum embed + pos_embed into one "embedding" term.
    embedding = cache["hook_embed"] + cache["hook_pos_embed"]   # (B, S, d_model)

    # Stack: 1 embed + 6 heads + 3 mlps = 10 components.
    out = [embedding]
    for layer in range(3):
        head_out = cache[utils.get_act_name("result", layer)]   # (B, S, n_heads, d_model)
        for head in range(2):
            out.append(head_out[:, :, head, :])
        out.append(cache[utils.get_act_name("mlp_out", layer)])
    return t.stack(out, dim=0)   # (10, B, S, d_model)
```

### Contribution to unbalanced direction

I project each component onto `pre_final_ln_dir` at position 0 only, subtract the mean over balanced examples (to make "balanced behavior" the zero baseline), then plot histograms per component split by true label (balanced vs unbalanced).

```python
out = get_out_by_components(model, data)[..., 0, :]                # keep seq_pos=0
score = einops.einsum(out, pre_final_ln_dir,                        # (10, B)
                      "comp batch d, d -> comp batch")

# Center by the mean on balanced examples.
score = score - score[:, data.is_balanced].mean(dim=1, keepdim=True)
```

Result: only **head 2.0**, **MLP 0**, **MLP 1**, and **MLP 2** have a bimodal-ish distribution split by label. **Head 2.0 dominates**. Heads in earlier layers barely contribute directly.

### Splitting by failure mode
I also split the unbalanced examples into two subtypes:
- `total_elevation_failure`: unequal counts of `(` vs `)`. Examples: `(()`, `())`, `((()`, ...
- `negative_failure`: reading right-to-left, we ever accumulate more `(` than `)`. Equivalently: some prefix (reading left-to-right) has more closes than opens. Examples: `())(`, `)(`, ...

```python
def classify_failure(toks_str):
    n_open  = toks_str.count("(")
    n_close = toks_str.count(")")
    total_elevation_failure = (n_open != n_close)

    # Scan right-to-left, treating "(" as +1 and ")" as -1.
    # Negative failure means some open bracket has no closer to its right.
    changes = [+1 if c == "(" else -1 for c in reversed(toks_str)]
    altitude = np.cumsum(changes)
    negative_failure = (altitude.max() > 0)

    return total_elevation_failure, negative_failure
```

Head 2.0 turns out to be almost entirely responsible for the **total_elevation_failure** cases. The **negative_failure** cases are handled by a different sub-circuit (some early heads + MLPs; not fully reverse-engineered in this notebook).

---

## 4. Head 2.0's role: reading from position 1

### Its attention pattern is trivial
When the query is at position 0, head 2.0's attention distribution puts almost all its weight on **position 1**. It ignores every other position.

```python
def get_attn_probs(model, data, layer, head):
    return get_activation(model, data.toks, utils.get_act_name("pattern", layer))[:, head, :, :]

attn_20 = get_attn_probs(model, data, layer=2, head=0)   # (B, S_q, S_k)
# For query=0: attn_20[:, 0, :] is ~1.0 at position 1 and ~0 elsewhere.
```

So head 2.0 basically implements: `out[0] = W_OV @ resid[1]`, up to a normalisation constant. The information used for classification MUST already be present at position 1 before layer 2. All the interesting computation happens upstream.

### Reversing through the OV circuit
Now I ask: what direction in resid[1] does head 2.0's W_OV send to `pre_final_ln_dir` at position 0?

```python
def get_WOV(model, layer, head):
    return model.W_V[layer, head] @ model.W_O[layer, head]   # (d_model, d_model)

def get_pre_20_dir(model, data):
    W_OV        = get_WOV(model, 2, 0)
    # Fit LN2 at seq_pos=1 (that's what head 2.0 reads).
    layer2_ln_fit, r2 = get_ln_fit(model, data, layernorm=model.blocks[2].ln1, seq_pos=1)
    L2          = t.from_numpy(layer2_ln_fit.coef_).to(device)
    post_ln_dir = get_pre_final_ln_dir(model, data)
    # Propagate the unbalanced direction backwards through:
    #   pre_final_ln_dir  <-  W_OV.T  <-  L2.T
    return L2.T @ W_OV @ post_ln_dir
```

`pre_20_dir` is now the direction such that "a component writing to resid[1] in this direction pushes the classifier toward unbalanced". Repeat the per-component histogram analysis, but at position 1 this time, over the 7 components that come before head 2.0 (embed + heads/MLPs of layers 0 and 1).

Result: the components that write most into `pre_20_dir` are **head 0.0**, **MLP 0**, and **MLP 1**. The other early heads and MLP 2 contribute little.

---

## 5. MLPs as key-value memories

Both MLP 0 and MLP 1 have neurons that fire strongly on unbalanced sequences at seq_pos=1. I look at individual neurons to see what functions they compute.

### The MLP-as-key-value view
For an MLP with `d_mlp` hidden neurons:
```
MLP(x)  =  sum over i in [d_mlp] of  f(x . W_in[:, i] + b_in[i])  *  W_out[i, :]
```
Each neuron `i` has:
- an **input direction** `W_in[:, i]` (the "key"),
- an **output direction** `W_out[i, :]` (the "value").

The neuron fires when the input activates its key direction, and when it does, it writes its value direction to the residual.

### Per-neuron output vectors

```python
def get_out_by_neuron(model, data, layer, seq=None):
    """
    Return the vector each neuron writes to the residual stream.
    Shape: (batch, [seq], neuron, d_model)
    """
    W_out = model.W_out[layer]                                 # (d_mlp, d_model)
    f_x_Win = get_activation(model, data.toks,                 # (B, S, d_mlp)
                             utils.get_act_name("post", layer))
    if seq is not None:
        f_x_Win = f_x_Win[:, seq, :]
    # (batch, [seq], neuron) * (neuron, d_model)  -> (batch, [seq], neuron, d_model)
    return einops.einsum(f_x_Win, W_out, "... n, n d -> ... n d")
```

### Interpreting individual neurons
For each neuron in MLP 0 and MLP 1, I compute its output projected onto `pre_20_dir` and plot against the sequence's open-proportion (fraction of open brackets among all brackets, ignoring padding).

- Some neurons fire strongly when **open-proportion > 0.5** (unbalanced-toward-opens).
- Others fire strongly when **open-proportion < 0.5** (unbalanced-toward-closes).
- Balanced sequences (open-proportion = 0.5) sit near the mode transition.

So the MLPs are computing **something like a threshold detector on the open-proportion**. They're the nonlinear step that turns "elevation = 0" vs "elevation != 0" into a signal that head 2.0 can then copy.

---

## 6. Head 0.0's role: computing the elevation

Now I work FORWARDS from the input to figure out where the open-proportion signal comes from.

### Head 0.0's attention pattern
Query the model with strings whose first bracket is `(`. Extract attention pattern of head 0.0. The pattern shows that **when the query token is `(` at position 1**, the head attends **approximately uniformly** to all key positions after it (up to the end of the sequence, ignoring pad).

```python
def get_q_and_k_for_given_input(model, tokenizer, parens_str, layer):
    q = get_activation(model, tokenizer.tokenize(parens_str), utils.get_act_name("q", layer))[0]
    k = get_activation(model, tokenizer.tokenize(parens_str), utils.get_act_name("k", layer))[0]
    return q, k   # each: (seq, n_heads, d_model)

# Activation patching: swap Q or K to see which one drives the uniform-attention behavior.
```

Uniform attention over positions >= 1 means: at query pos 1, the head is essentially **averaging its OV output over the entire remaining suffix**.

### The OV circuit is one-dimensional
Compute the two "value-then-output" vectors:
```python
W_OV_00 = model.W_V[0, 0] @ model.W_O[0, 0]

L0 = get_ln_fit(model, data, model.blocks[0].ln1, seq_pos=None)[0]
L0 = t.from_numpy(L0.coef_).to(device)

v_L = model.W_E[tok_id("(")] @ L0.T @ W_OV_00     # vector written when the KEY is "("
v_R = model.W_E[tok_id(")")] @ L0.T @ W_OV_00     # vector written when the KEY is ")"
```

Cosine similarity of `v_L` and `v_R` is approximately -1. So the OV output is essentially **`+v_R` per `)` token and `-v_R` per `(` token**. Averaged uniformly over the suffix, this gives:
```
head_0.0_output[1]  proportional to  (n_close - n_open) / suffix_length
                    proportional to  (0.5 - open_proportion)      <-- the "elevation"
```

That's the total elevation signal. Head 0.0 computes it in one shot.

---

## 7. The full circuit (working forwards)

Putting all seven exercises together:

1. **Embed + pos_embed**: each token position gets its bracket-identity signal (via `W_E`) + a positional signal (sinusoidal).
2. **Head 0.0** at position 1: uniform attention over the whole suffix, W_OV maps each `(` to `-v` and each `)` to `+v`. Averaged, this writes `(open_proportion - 0.5) * v` to `resid[1]`.
   - This is the **elevation signal** encoded in one residual-stream direction.
3. **MLP 0 and MLP 1** at position 1: apply nonlinear thresholding. Individual neurons fire when elevation is positive vs negative. Aggregated, they encode "the elevation is NOT zero" into a direction in `resid[1]`.
4. **Head 2.0** at position 0: attends only to position 1. W_OV copies the MLP-processed signal into `resid[0]`.
5. **Final LN + W_U + softmax**: reads `resid[0]` in the unbalanced direction. High dot product -> classify unbalanced.

The elegance: head 2.0's attention pattern is degenerate on purpose. All the model needs to do is aggregate the elevation into one position (position 1), threshold it (MLPs), and read it out (head 2.0).

---

## 8. Bonus - other failure modes

The main circuit only handles **total-elevation failures**. Cases like `())(` have equal counts of `(` and `)` (so the head-0.0-style elevation signal is 0) but are still unbalanced because a `)` appeared with no matching `(` on its left.

### Detecting anywhere-negative failures
The negative-failure detector must live somewhere else. Candidate sub-circuits:
- Head 0.1 or heads in layer 1 could compute a running "max prefix-elevation" via more targeted attention.
- MLP 2 might also participate.

The notebook lightly explores this: plotting attention patterns of every head over failure-vs-non-failure cases, and looking for heads that discriminate on `()`-order rather than count.

### Adversarial attacks
Once we know the model roughly does "count opens minus closes, then check for negative", we can construct adversarial inputs that fool it. Long sequences of matched brackets padded with a single mismatched pair at the end can slip past the count check if the mismatch is exactly cancelled elsewhere. The notebook demonstrates a few such attacks.

### Dealing with early closing parens
If a sequence starts with a `)`, that's an immediate negative failure, but head 0.0's uniform-suffix attention only fires for `(` at position 1. So there must be a separate detector for "position 1 starts with `)`". Simple candidate: a direction in the embedding of `)` at position 1 already flags this via the direct-embed path (embedding + pos_embed contributes directly to `resid_pre_final`).

---

## Takeaways (across all exercises)

- **Reverse-engineering is a back-and-forth**: I worked BACKWARDS from the logits (through softmax, W_U, LN, W_OV) to find "unbalanced directions" in each stage of the residual stream, then FORWARDS from the input to explain how those directions get set.
- **LayerNorm can be linearised** when the input distribution is stable. Fit a linear regression from LN-input to LN-output; R^2 near 1 means the trick works. This is the key that lets us treat the whole classifier tail as a linear map for the purpose of finding directions.
- **Attention heads compose modularly with MLPs via the residual stream**. Head 0.0 writes an elevation signal, MLPs threshold it, head 2.0 copies the result. No single component does the whole job.
- **A near-degenerate attention pattern is a big clue**: when head 2.0 attends only to one position, we know its function is essentially "move info from A to B". The interesting work happens somewhere else.
- **The OV circuit's dimensionality tells you what the head is doing**: head 0.0's OV maps `(` and `)` to approximately antiparallel vectors, meaning it's projecting onto a single direction - the elevation.
- **Direct-logit-attribution works for classification too**: it's the same "project each component's output onto a target direction" trick used for GPT-2 next-token prediction, just applied to a 2-class classification head.
- **This tiny model uses about half its capacity for the main circuit** (the elevation path). The rest is other sub-circuits (negative-failure detection, edge cases) that we didn't fully reverse-engineer.
