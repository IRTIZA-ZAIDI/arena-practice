# TransformerLens cheat sheet

A short reference for the bits of `transformer_lens` I keep using across these exercises (mostly [1.2 mech interp](../intro_to_mech_interp/notes.md) and [1.4.1 IOI](notes.md)). Shapes, hook names, common helpers. Read top to bottom for the API; or jump to the section matching the operation at hand.

```python
from transformer_lens import HookedTransformer, ActivationCache, FactoredMatrix, utils, patching
from transformer_lens.hook_points import HookPoint
import torch as t, einops
```

---

## 1. Loading a model

```python
model = HookedTransformer.from_pretrained("gpt2-small")
```

Important flags (defaults shown):

| Flag | Default | What it does |
|---|---|---|
| `center_unembed` | `True` | Sets `W_U` to be mean-zero per column. Numerically equivalent (logits get a constant subtracted), but makes "direct logit attribution" clean. |
| `center_writing_weights` | `True` | Mean-centers the columns of write-out matrices (`W_O`, MLP `W_out`, embeddings). Same idea. |
| `fold_ln` | `True` | Folds the learned LayerNorm scale + bias into the next linear layer. After folding, the only LN nonlinearity is `hook_scale` (a scalar per token), which is constant on a fixed input -> logit attribution becomes exactly additive. |
| `refactor_factored_attn_matrices` | `False` | Rotates `W_Q` / `W_K` (and `W_V` / `W_O`) to share singular-value bases so individual head columns are orthogonal + equal-norm. Easier per-head interp. |

**When to use which**: for casual exploration, defaults are fine. For numerical-equivalence reimplementation (Transformers from Scratch), pass all `False`. For circuit analysis (IOI), set all `True` plus `refactor_factored_attn_matrices=True`.

---

## 2. Tokenization

`model.tokenizer` is the underlying HF tokenizer. TransformerLens adds:

```python
tokens = model.to_tokens(text, prepend_bos=True)     # (1, seq) int64
ids    = model.to_tokens([s1, s2], prepend_bos=True) # batched
strs   = model.to_str_tokens(text)                   # ["<|endoftext|>", " When", " John", ...]
strs   = model.to_str_tokens(tokens)                 # also accepts tensors
text   = model.to_string(tokens)                     # decode back to string
single = model.to_single_token(" Mary")              # int, asserts the string is exactly one token
```

Notes:
- `prepend_bos=True` adds `<|endoftext|>` (token 50256) at position 0. Most pretrained TL models were trained with it.
- `to_str_tokens` is for visualization, not for indexing. Spaces are usually preserved (`" John"`, not `"John"`).
- For `to_single_token`, leading space matters. `" Mary"` is one token; `"Mary"` is often two.

---

## 3. Running the model

Five ways to call:

```python
# 1. Plain forward: returns logits (batch, seq, d_vocab).
logits = model(tokens)

# 2. With a custom return type.
loss = model(tokens, return_type="loss")                 # scalar cross-entropy
both = model(tokens, return_type="both")                 # (logits, loss)
nothing = model(tokens, return_type=None)                # skip the final unembed, save compute

# 3. With cache. Returns (logits, ActivationCache).
logits, cache = model.run_with_cache(tokens)

# 4. With forward hooks. Hooks are (name, fn) pairs.
logits = model.run_with_hooks(
    tokens,
    return_type="logits",
    fwd_hooks=[("blocks.5.attn.hook_pattern", my_hook_fn)],
)

# 5. Both.
logits, cache = model.run_with_cache(
    tokens,
    fwd_hooks=[(name_filter_lambda, my_hook_fn)],
)
```

**`stop_at_layer=N`** (kwarg on `run_with_cache`): runs only the first N layers. Huge speedup when only early-layer activations are needed:

```python
_, cache = model.run_with_cache(tokens, stop_at_layer=7)   # don't compute past layer 6
```

**`names_filter`** (kwarg): a callable or list. Only those hook names get cached. Critical for memory:

```python
_, cache = model.run_with_cache(
    tokens,
    names_filter=lambda name: name.endswith("pattern"),
)
```

---

## 4. The ActivationCache

A dict mapping hook names to tensors, with helpers.

### Shorthand indexing

```python
cache["pattern", 0]                # attention pattern at layer 0
cache["q", 0]                      # query vectors at layer 0
cache["k", 0]                      # key vectors at layer 0
cache["v", 0]                      # value vectors at layer 0
cache["z", 0]                      # post-attention z (mixed values, before W_O)
cache["result", 0]                 # attn output per head, before sum + W_O. NEEDS use_attn_result=True in cfg.
cache["attn_out", 0]               # final attn output (after summing heads + W_O)
cache["mlp_out", 0]                # MLP output
cache["resid_pre", 0]              # residual stream before layer 0's attn
cache["resid_mid", 0]              # between attn and MLP
cache["resid_post", 0]             # after layer 0's MLP. Same as resid_pre of layer 1.
cache["embed"]                     # token embedding (no layer index)
cache["pos_embed"]                 # positional embedding
cache["normalized", 0, "ln1"]      # after layer 0's first LN
cache["scale", 0, "ln1"]           # the scalar scale from layer 0's LN1
```

The shorthand `(activation_name, layer)` resolves to `utils.get_act_name(name, layer)` -> a full string like `"blocks.0.attn.hook_pattern"`.

### Common shapes (batch=B, seq=S, d_model=D, n_heads=H, d_head=D_H, n_layers=L)

| Cache key | Shape |
|---|---|
| `embed`, `pos_embed` | `(B, S, D)` |
| `resid_pre`, `resid_mid`, `resid_post` | `(B, S, D)` |
| `attn_out`, `mlp_out` | `(B, S, D)` |
| `q`, `k`, `v` | `(B, S, H, D_H)` |
| `z` | `(B, S, H, D_H)` |
| `pattern` | `(B, H, S, S)` (softmaxed) |
| `attn_scores` | `(B, H, S, S)` (pre-softmax) |
| `result` (needs `use_attn_result=True`) | `(B, S, H, D)` |
| `scale` | `(B, S, 1)` |

### Useful cache helpers

```python
# Cumulative residual after each component (embed, attn_0, mlp_0, attn_1, ...).
# Shape: (n_components, batch, d_model) when pos_slice is a scalar.
acc, labels = cache.accumulated_resid(
    layer=-1,                       # up to which layer (-1 = last)
    incl_mid=True,                  # include resid_mid points
    pos_slice=-1,                   # slice positions; -1 = last position only
    return_labels=True,
)


# Per-component decomposition (embed + attn_l + mlp_l for each l).
# Shape: (n_components, batch, d_model).
per_layer, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)


# Per-head decomposition (one row per (layer, head)).
# Shape: (n_layers * n_heads, batch, d_model).
per_head, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
per_head = einops.rearrange(per_head, "(L H) ... -> L H ...", L=model.cfg.n_layers)


# Apply the final LayerNorm scale to a stack of components.
# Use this before projecting onto a logit-direction; otherwise per-component
# contributions don't sum to the actual logit value.
scaled = cache.apply_ln_to_stack(stack, layer=-1, pos_slice=-1)
```

### Other cache utilities

```python
cache.remove_batch_dim()         # in-place: squeezes the batch dim (must be size 1)
list(cache.keys())               # all stored hook names
"blocks.0.attn.hook_pattern" in cache
cache.items()                    # iterate (name, tensor) pairs
```

---

## 5. The hook system

Every interesting activation has a *hook point*, addressable by name. Hooks are forward callbacks: `(activation, hook) -> activation_or_None`. Returning a modified tensor replaces the activation; returning None just observes.

```python
def my_hook(activation, hook):
    # `activation` is the tensor at this site
    # `hook` is a HookPoint object with .name, .layer(), .ctx (a dict for storage)
    activation[:, 5, :] = 0     # in-place modify
    return activation           # or return None to not replace
```

### `run_with_hooks` (one-shot)

```python
logits = model.run_with_hooks(
    tokens,
    return_type="logits",
    fwd_hooks=[
        ("blocks.5.attn.hook_pattern", my_hook),               # specific hook
        (lambda name: "hook_q" in name, my_other_hook),         # filter all matching hooks
    ],
)
```

`run_with_hooks` automatically clears hooks afterwards. Multiple hooks on the same name fire in registration order.

### Persistent hooks

For sticky hooks across multiple forward passes:

```python
model.add_hook("blocks.5.attn.hook_pattern", my_hook)   # fires on every forward
model.run_with_cache(tokens)                            # hook is active
model.reset_hooks()                                     # remove all permanently-added hooks
```

### Hook names: `utils.get_act_name`

```python
utils.get_act_name("pattern", 0)     # "blocks.0.attn.hook_pattern"
utils.get_act_name("z", 5)           # "blocks.5.attn.hook_z"
utils.get_act_name("resid_pre", 3)   # "blocks.3.hook_resid_pre"
utils.get_act_name("mlp_out", 11)    # "blocks.11.hook_mlp_out"
utils.get_act_name("embed")          # "hook_embed"
```

Use this instead of hand-writing hook strings; the function knows the conventions for every site.

### Hooks: writing vs reading

```python
# Read-only (cache to a global dict).
results = {}
def read_hook(act, hook):
    results[hook.name] = act.detach().cpu()
    # No return -> activation is unchanged.

# Write (intervention).
def zero_head_hook(act, hook, head_idx):
    act[:, :, head_idx, :] = 0       # in-place modification of z
    return act

from functools import partial
hook = partial(zero_head_hook, head_idx=4)
model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("z", 5), hook)])
```

### Parametrizing hooks with `functools.partial`

The hook signature is fixed: `(activation, hook) -> activation_or_None`. But most real interventions need extra arguments (`head_idx`, `seq_pos`, `clean_cache`, `sender_layer`, ...). `partial` bakes those in so the resulting callable still matches the expected signature.

The pattern shows up everywhere in the IOI exercises - any time the same hook function gets reused across many `(layer, head)` or `(layer, position)` combinations in a sweep.

**Single extra argument: zero one head per sweep step**

```python
from functools import partial

def head_zero_ablation_hook(z, hook, head_index_to_ablate):
    z[:, :, head_index_to_ablate, :] = 0
    return z

for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        hook_fn = partial(head_zero_ablation_hook, head_index_to_ablate=head)
        patched_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(utils.get_act_name("z", layer), hook_fn)],
        )
        # ... score patched_logits
```

**Multiple extra arguments: activation patching with a clean cache**

```python
def patch_residual_component(corrupted_resid, hook, pos, clean_cache):
    corrupted_resid[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_resid

# Bake in the position AND the clean cache; the resulting fn is a valid hook.
for layer in range(model.cfg.n_layers):
    for pos in range(tokens.shape[1]):
        hook_fn = partial(patch_residual_component, pos=pos, clean_cache=clean_cache)
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(utils.get_act_name("resid_pre", layer), hook_fn)],
        )
```

**Path-patching style: freeze every head except one**

```python
def patch_or_freeze_head_vectors(z, hook, new_cache, orig_cache, head_to_patch):
    # Step 1: freeze ALL heads to the orig (clean) values.
    z[...] = orig_cache[hook.name][...]
    # Step 2: patch the single sender head with the new (corrupted) value.
    layer, head = head_to_patch
    if hook.layer() == layer:
        z[:, :, head, :] = new_cache[hook.name][:, :, head, :]
    return z

# The SAME partial gets attached to every layer's "z" hook in the sweep.
hook_fn = partial(
    patch_or_freeze_head_vectors,
    new_cache=corrupted_cache, orig_cache=clean_cache,
    head_to_patch=(sender_layer, sender_head),
)
fwd_hooks = [(utils.get_act_name("z", L), hook_fn) for L in range(model.cfg.n_layers)]
```

**Why not closures?** A `def` nested in a loop captures the loop variable by reference, so all the closures end up sharing the *final* value (the classic "late binding" bug). `partial` evaluates its args eagerly, so each iteration gets its own bound copy. `lambda head=head: hook(head)` also works but reads worse.

**Why not class-based hooks?** They work (and `DeceptionSteeringHook` in linear_probe uses one for the enable/disable lifecycle). But for one-shot sweeps inside `run_with_hooks`, `partial` is one line and has no state to manage.

**Tip**: use keyword args, not positional, when calling `partial`. `partial(fn, head_idx=4)` is explicit and survives signature changes; `partial(fn, 4)` silently breaks when someone adds a parameter.

### `hook.ctx` for storage between hooks

```python
def cache_to_ctx(act, hook):
    hook.ctx["my_data"] = act[:, -1, :].clone()

# Later, after the forward:
saved = model.blocks[5].attn.hook_pattern.ctx["my_data"]
```

---

## 6. Model parameters and shapes

All weights are accessible as `nn.Parameter`s:

```python
model.W_E              # (d_vocab, d_model)              embedding
model.W_pos            # (n_ctx,   d_model)              positional embedding (if learned)
model.W_U              # (d_model, d_vocab)              unembedding
model.b_U              # (d_vocab,)                      unembed bias (often frozen at 0)

model.W_Q              # (n_layers, n_heads, d_model, d_head)
model.W_K              # same
model.W_V              # same
model.W_O              # (n_layers, n_heads, d_head, d_model)
model.b_Q, model.b_K, model.b_V                 # (n_layers, n_heads, d_head)
model.b_O                                       # (n_layers, d_model)

model.W_in             # (n_layers, d_model, d_mlp)      MLP up-projection
model.W_out            # (n_layers, d_mlp, d_model)      MLP down-projection
model.b_in, model.b_out

model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_model, model.cfg.d_head
model.cfg.d_mlp, model.cfg.d_vocab, model.cfg.n_ctx
```

Single-head indexing: `model.W_Q[layer, head]` gives `(d_model, d_head)`.

---

## 7. The patching module

Standardized activation-patching sweeps. Saves a lot of boilerplate.

```python
from transformer_lens import patching

# Per (layer, position) heatmap for resid_pre. Shape: (n_layers, seq).
patching.get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, metric)

# Faceted by sublayer. Shape: (3, n_layers, seq). Facets: resid / attn_out / mlp_out.
patching.get_act_patch_block_every(model, corrupted_tokens, clean_cache, metric)

# Per-head, all positions. Shape: (n_layers, n_heads).
patching.get_act_patch_attn_head_out_all_pos(model, corrupted_tokens, clean_cache, metric)

# Per-head, per-position. Shape: (n_layers, seq, n_heads).
patching.get_act_patch_attn_head_out_by_pos(model, corrupted_tokens, clean_cache, metric)

# Patch q/k/v/pattern instead of z output.
patching.get_act_patch_attn_head_q_all_pos(model, corrupted_tokens, clean_cache, metric)
patching.get_act_patch_attn_head_k_all_pos(...)
patching.get_act_patch_attn_head_v_all_pos(...)
patching.get_act_patch_attn_head_pattern_all_pos(...)
```

Signature is consistent: `(model, corrupted_tokens, clean_cache, metric_fn)` where `metric_fn(logits) -> scalar`.

For path patching (which the library doesn't ship a one-shot helper for), see the IOI notes - it's a 3-step manual sweep with `model.run_with_hooks`.

---

## 8. Useful idioms

### Get the residual direction of specific tokens

```python
# Shape: (batch, 2, d_model) for two tokens per prompt.
answer_directions = model.tokens_to_residual_directions(answer_tokens)
# Convention: tokens_to_residual_directions(t) returns W_U[:, t].T (after centering).
```

### Project a residual-stream tensor onto the unembed for one specific token

```python
logit_for_token = (resid @ model.W_U)[:, token_id]
# Equivalently:
logit_for_token = resid @ model.W_U[:, token_id]
```

### Apply final LayerNorm scale before logit attribution

```python
# WRONG: per-component contributions won't sum to the actual logit.
attn_l9_contribution = (cache["result", 9].sum(dim=-2) @ logit_direction)

# RIGHT: divide by the SAME final-LN scale that the full residual saw.
scaled = cache.apply_ln_to_stack(cache["result", 9].sum(dim=-2), layer=-1, pos_slice=-1)
attn_l9_contribution = scaled @ logit_direction
```

### Causal mask + softmax over key positions (hand-recomputing attention)

```python
q, k = cache["q", 0], cache["k", 0]               # (S, H, D_H)
seq, _, head_size = q.shape

scores = einops.einsum(q, k, "sQ H d, sK H d -> H sQ sK")
mask = t.triu(t.ones(seq, seq, dtype=t.bool, device=device), diagonal=1)
scores.masked_fill_(mask, -1e9)
pattern = (scores / head_size**0.5).softmax(-1)
```

### Stop the forward early after capturing the needed activations

```python
class StopForward(Exception): pass

def stop_hook(act, hook):
    raise StopForward()

try:
    model.run_with_hooks(tokens, fwd_hooks=[("blocks.10.hook_resid_pre", stop_hook)])
except StopForward:
    pass
```

This is the standard trick when only shallow-layer activations are needed and the rest of the forward can be skipped.

### Build a name filter for batched caching

```python
# Cache only attention patterns, nothing else.
pattern_filter = lambda name: name.endswith("hook_pattern")
_, cache = model.run_with_cache(tokens, names_filter=pattern_filter)

# Cache only the layers of interest.
layers_to_cache = {5, 6, 7}
my_filter = lambda name: any(f"blocks.{L}." in name for L in layers_to_cache)
```

### Per-head splits and concatenations

```python
# z is (B, S, H, D_H). To recover its residual-stream contribution per head:
# Reshape W_O[layer] from (H, D_H, D) and contract.
per_head_attn_out = einops.einsum(
    cache["z", layer], model.W_O[layer],
    "B S H D_H, H D_H D -> B S H D",
)
# This is what cache["result", layer] would give (with use_attn_result=True),
# but constructed manually.
```

### FactoredMatrix - lazy products

```python
from transformer_lens import FactoredMatrix

W_OV = FactoredMatrix(model.W_V[layer, head], model.W_O[layer, head])
W_EQK = FactoredMatrix(model.W_E @ model.W_Q[layer, head],
                       model.W_E @ model.W_K[layer, head]).T   # (d_vocab, d_vocab)

# Never materializes the (50k, 50k) product. Supports:
W_EQK.norm()
W_EQK.svd()             # returns (U, S, V) of the underlying product
W_EQK[i, j]             # scalar lookup
W_EQK[i, :]             # row slice (still a FactoredMatrix)
W_EQK.AB                # force-materialize (use only on a slice)
```

Essential for `(d_vocab, d_vocab)`-shaped circuit matrices that would otherwise be 10 GB.

---

## 9. Common gotchas

- **`fold_ln=True` changes the layer-norm parameters into the next linear layer's bias/weights.** After folding, `model.blocks[i].ln1.w` is a tensor of ones and `model.blocks[i].ln1.b` is zeros - the meaningful scale + bias have moved into the attention's `W_Q`/`W_K`/`W_V`. This is what enables additive logit attribution.
- **`use_attn_result=False` by default** to save memory. To get `cache["result", layer]` (per-head output), pass `model.cfg.use_attn_result = True` before `run_with_cache`. Otherwise reconstruct manually from `z` and `W_O` as shown above.
- **Caching everything is expensive.** For a 12L GPT-2 with 8 prompts of seq 32, a full cache is ~50 MB. For Llama-3.1-8B on long sequences it's gigabytes. Always pass `names_filter` and/or `stop_at_layer` when possible.
- **`pos_slice=-1` vs `pos_slice=None`** in cache helpers: the former takes one position (last token), shrinking the position axis to a scalar. The latter keeps all positions.
- **Hook outputs that are tuples**: some HuggingFace blocks return `(hidden, ...)` tuples. TransformerLens normalizes this; the internal hook plumbing handles unpacking, so it usually doesn't come up in user code.
- **`prepend_bos=True` is the default for `to_tokens`**. Forgetting it means the position-0 representation is a real token rather than BOS, and many trained-with-BOS models behave noticeably worse.
- **`refactor_factored_attn_matrices=True` changes Q/K/V/O weights** but not the model's behavior. Tests that compare numerical output don't need to special-case it; tests that introspect specific weight values do.

---

## 10. Standard workflow templates

### Logit attribution

```python
logits, cache = model.run_with_cache(tokens)
logit_diff_direction = model.W_U[:, IO_token] - model.W_U[:, S_token]   # (d_model,)

per_head, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
per_head = einops.rearrange(per_head, "(L H) B D -> L H B D", L=model.cfg.n_layers)
scaled = cache.apply_ln_to_stack(per_head, layer=-1, pos_slice=-1)
per_head_logit_diff = einops.einsum(scaled, logit_diff_direction,
                                     "L H B D, D -> L H B").mean(-1)
# (L, H) heatmap
```

### Activation patching (head -> output)

```python
from transformer_lens import patching

_, clean_cache = model.run_with_cache(clean_tokens)

def metric(logits):
    return (logits_to_logit_diff(logits) - corrupt_diff) / (clean_diff - corrupt_diff)

heatmap = patching.get_act_patch_attn_head_out_all_pos(
    model, corrupted_tokens, clean_cache, metric,
)
# (L, H) heatmap of patching effect
```

### Custom hook intervention

```python
def patch_head(act, hook, head_idx, clean_cache):
    # act shape: (B, S, H, D_H) for "z" hook
    act[:, :, head_idx, :] = clean_cache[hook.name][:, :, head_idx, :]
    return act

hook = partial(patch_head, head_idx=9, clean_cache=clean_cache)
patched_logits = model.run_with_hooks(
    corrupted_tokens,
    fwd_hooks=[(utils.get_act_name("z", 9), hook)],
)
```

### Visualizing attention

```python
import circuitsvis as cv
from IPython.display import display

display(cv.attention.attention_patterns(
    tokens=model.to_str_tokens(text),
    attention=cache["pattern", layer],          # (H, S, S) - already squeezed batch
    attention_head_names=[f"L{layer}H{h}" for h in range(model.cfg.n_heads)],
))
```

---

## See also

- [ARENA TransformerLens Demo notebook](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Main_Demo.ipynb) for the canonical examples.
- [TransformerLens docs](https://transformerlensorg.github.io/TransformerLens/) for the full hook-name list and config options.
- Local notes that exercise this API:
  - [intro_to_mech_interp/notes.md](../intro_to_mech_interp/notes.md) for the basics (cache, hooks, induction heads).
  - [indirect_object_identification/notes.md](notes.md) for the heavy circuit-level use (logit attribution, activation patching, path patching).
  - [interpretability_with_SAEs/notes.md](../interpretability_with_SAEs/notes.md) for the SAELens extension that adds `HookedSAETransformer`.
