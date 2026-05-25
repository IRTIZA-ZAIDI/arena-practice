# Prerequisites - Notes

Notes from ARENA [0.0] Prerequisites. Companion to [0_0_Prerequisites_exercises.ipynb](0_0_Prerequisites_exercises.ipynb).

Scope: getting fluent with `einops`, `einsum`, broadcasting, indexing, and the numerical patterns (logsumexp / softmax / cross-entropy) that show up everywhere downstream.

---

## 1. Mental model for einops

Three primitives, all driven by an axis-pattern string:

- **`rearrange`**: reshape/transpose only. No reduction, no copy. `c h w -> c w h`, `b c h w -> c h (b w)`.
- **`repeat`**: like `rearrange` but introduces *new* axes that get tiled.
- **`reduce`**: collapse axes with `"mean"`, `"sum"`, `"max"`, or a torch function like `t.std`.

Two grouping patterns to internalize:
- `"(h 2)"` = repeat *each* `h` element twice in place: `[A,B,C] -> [A,A,B,B,C,C]` (stretching).
- `"(2 h)"` = repeat the *whole* `h` sequence twice: `[A,B,C] -> [A,B,C,A,B,C]` (tiling).

The factor on the **left** means "outer loop"; on the **right** means "inner loop". This single distinction killed half the bugs in the image exercises.

### Image manipulation snippets

`arr` has shape `(b, c, h, w)` (batch of digit images).

```python
# Column-stacking: 6 digits side by side. Merge batch into width.
arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")            # stack vertically
arr_row = einops.rearrange(arr, "b c h w -> c h (b w)")         # stack horizontally

# Copy the first digit twice vertically (one image -> doubled height).
arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")

# First 2 digits, stack vertically and double the width by repeating.
arr3 = einops.repeat(arr[:2], "b c h w -> c (b h) (2 w)")

# Stretch a single image vertically by a factor of 2 (each row repeated).
arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")

# Split channels of one image into 3 horizontal monochrome panels.
arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")

# 6-digit batch into a 2x3 grid using a factored batch axis.
arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)

# Transpose H<->W on one image.
arr7 = einops.rearrange(arr[1], "c h w -> c w h")

# Downsample 2x2 by max-pooling, then arrange into a grid.
arr8 = einops.reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max",
                     h2=2, w2=2, b1=2)
```

### Learning
- Whenever you write `(a b)` in einops, decide if you mean "outer a, inner b" or vice versa. Get this wrong and your image will be visibly wrong (interleaved instead of tiled), which is a great fast feedback loop.
- `reduce` with a factored axis pattern is how you do pooling. The trick: pull out the pool window as its own axis (e.g. `(h h2)` with `h2=2`), then reduce over it.

---

## 2. Broadcasting

Rules (NumPy and PyTorch, identical):
1. Pad missing leading dims with size 1.
2. Two dims are compatible if equal, or one is 1.
3. Size-1 dims get virtually expanded to the matching size.

The two shape-fixers you need most:
- `tensor.unsqueeze(dim)` adds a size-1 axis.
- `tensor.squeeze(dim)` removes a size-1 axis.

For the temperature exercises, `temps` has shape `(14,)` and the per-week mean is `(2,)`. These don't broadcast. Two ways to fix:
- Repeat the means back to `(14,)`: `einops.repeat(avg, "w -> (w 7)")`.
- Or reshape `temps` to `(2, 7)`, subtract, then flatten.

### Snippets

```python
# Weekly mean: collapse the 7-day axis with reduce.
def temperatures_average(temps: Tensor) -> Tensor:
    return einops.reduce(temps, "(h 7) -> h", "mean")

# Subtract the weekly mean from each day. Note (w 7) tiles each w 7 times.
def temperatures_differences(temps: Tensor) -> Tensor:
    avg = einops.reduce(temps, "(h 7) -> h", "mean")
    return temps - einops.repeat(avg, "w -> (w 7)")

# Z-score per week. t.std can be passed directly to reduce.
def temperatures_normalized(temps: Tensor) -> Tensor:
    avg = einops.reduce(temps, "(h 7) -> h", "mean")
    std = einops.reduce(temps, "(h 7) -> h", t.std)
    return (temps - einops.repeat(avg, "w -> (w 7)")) / einops.repeat(std, "w -> (w 7)")
```

---

## 3. Norms, cosine similarity, sampling

```python
# L2-normalize each row. keepdim=True is what makes broadcasting work.
def normalize_rows(matrix: Tensor) -> Tensor:
    norm = matrix.norm(dim=1, keepdim=True)
    return matrix / norm

# Pairwise cosine similarity: normalize then matmul.
# (normalized @ normalized.T)[i, j] = sum_k normalized[i,k] * normalized[j,k]
def cos_sim_matrix(matrix: Tensor) -> Tensor:
    normalized = normalize_rows(matrix)
    return normalized @ normalized.T

# Sample from a categorical without loops.
# Trick: cumsum -> CDF, then count how many CDF entries each uniform sample exceeds.
def sample_distribution(probs: Tensor, n: int) -> Tensor:
    cum = t.cumsum(probs, dim=0)        # shape (k,)
    rand = t.rand(n, 1)                 # shape (n, 1), broadcasts vs (k,)
    return (rand > cum).sum(dim=-1)     # count thresholds crossed = sampled index
```

### Learning
- `keepdim=True` is the difference between code that works and code that silently broadcasts wrong. Default to `keepdim=True` for any reduction you'll divide by.
- The `(rand > cum).sum()` sampler is a great pattern. No loops, no `torch.multinomial`. The boolean tensor expansion replaces explicit search.

---

## 4. Indexing & gather

Two interchangeable ways to look up values:

```python
# Integer array indexing (NumPy-style). prices[items] picks elements by index.
def total_price_indexing(prices: Tensor, items: Tensor) -> float:
    return prices[items].sum()

# torch.gather: per-position lookup along one axis.
# 1D version is equivalent to indexing.
def total_price_gather(prices: Tensor, items: Tensor) -> float:
    assert items.max() < prices.shape[0]
    return prices.gather(0, items).sum().item()

# gather_2d: out[i][j] = matrix[i][indexes[i][j]]. Indexes must have same ndim as matrix.
def gather_2d(matrix: Tensor, indexes: Tensor) -> Tensor:
    assert matrix.ndim == indexes.ndim
    assert indexes.shape[0] == matrix.shape[0]
    out = matrix.gather(1, indexes)
    assert out.shape == indexes.shape
    return out

# Multi-dim integer indexing: matrix[tuple(coords.T)] for arbitrary-rank tensors.
# coords shape (batch, n), matrix shape (d_0, ..., d_{n-1}), result shape (batch,).
def integer_array_indexing(matrix: Tensor, coords: Tensor) -> Tensor:
    return matrix[tuple(coords.T)]

# Row / column selection (these are just regular indexing).
def collect_rows(matrix, row_indexes):    return matrix[row_indexes]
def collect_columns(matrix, column_indexes): return matrix[:, column_indexes]
```

### When to use which
- `tensor[idx]` (integer array indexing): cleanest for "pick these rows / these positions / these coords". Works on multi-dim with `matrix[tuple(coords.T)]`.
- `tensor.gather(dim, idx)`: needed when the lookup is **per-row**, e.g. for each row pick a different column. The index tensor must have the same `ndim` as the source, with all dims matching except `dim`.

`gather` is what you reach for in cross-entropy loss (per-example pick the logit for the true class) and beam search.

### Learning
- If `idx.shape == src.shape` along all axes except one, you want `gather`. Otherwise you want integer indexing.

---

## 5. Numerical stability: logsumexp, softmax, cross-entropy

These are the same pattern repeated. **Subtract the per-row max before exp**, then add it back outside the log. Otherwise `exp(1000) -> inf`.

```python
# logsumexp(x) = max(x) + log(sum(exp(x - max(x)))).
def batched_logsumexp(matrix: Tensor) -> Tensor:
    C = matrix.max(dim=-1).values
    exps = t.exp(matrix - einops.rearrange(C, "n -> n 1"))
    return C + t.log(t.sum(exps, dim=-1))

# softmax: shift by max, exp, divide by sum. Translation-invariant by construction.
def batched_softmax(matrix: Tensor) -> Tensor:
    shift = matrix - matrix.max(dim=-1, keepdim=True).values
    exp = t.exp(shift)
    return exp / t.sum(exp, dim=-1, keepdim=True)

# logsoftmax: x - max - log(sum(exp(x - max))).
def batched_logsoftmax(matrix: Tensor) -> Tensor:
    C = matrix.max(dim=1, keepdim=True).values
    return matrix - C - (matrix - C).exp().sum(dim=1, keepdim=True).log()

# Cross-entropy: -logsoftmax at the true class.
def batched_cross_entropy_loss(logits: Tensor, true_labels: Tensor) -> Tensor:
    assert logits.shape[0] == true_labels.shape[0]
    assert true_labels.max() < logits.shape[1]
    logprobs = batched_logsoftmax(logits)
    indices = einops.rearrange(true_labels, "n -> n 1")
    pred_at_index = logprobs.gather(1, indices)
    return -einops.rearrange(pred_at_index, "n 1 -> n")

# Classifier accuracy: fraction where argmax matches the true label.
def classifier_accuracy(scores: Tensor, true_classes: Tensor) -> Tensor:
    return (t.argmax(scores, dim=-1) == true_classes).float().mean()
```

### Learning
- `tensor.max(dim=...)` returns a `(values, indices)` namedtuple. You almost always want `.values`. This bit me until I burned it in.
- Softmax is translation-invariant: `softmax(x) == softmax(x + c)` for any constant `c`. That's why the max-shift trick is exact, not an approximation.
- Cross-entropy loss = `logsoftmax` + `gather` + negate. Don't write your own `exp` + `log` chain; you'll lose precision.

---

## 6. Einsum

Use `einops.einsum(tensor1, tensor2, ..., "pattern1, pattern2, ... -> output")`. Note: spaces between axis names in `einops.einsum` (vs no spaces in `torch.einsum`).

```python
# Trace: force matching indices, sum along diagonal.
def einsum_trace(mat):
    return einops.einsum(mat, "i i ->")

# Matrix-vector product.
def einsum_mv(mat, vec):
    return einops.einsum(mat, vec, "i j, j -> i")

# Matrix-matrix product.
def einsum_mm(mat1, mat2):
    return einops.einsum(mat1, mat2, "i j, j k -> i k")

# Inner product (dot product).
def einsum_inner(vec1, vec2):
    return einops.einsum(vec1, vec2, "i, i ->")

# Outer product.
def einsum_outer(vec1, vec2):
    return einops.einsum(vec1, vec2, "i, j -> i j")
```

### Four rules for reading any einsum pattern
1. An axis in **input AND output**: kept.
2. An axis in **input only**: summed over.
3. **Same letter across tensors**: align/match that dim.
4. **Different letters**: outer-product style combination.

### Patterns worth memorizing

```python
# Sum all elements:            "i j ->"
# Row sums:                    "i j -> i"
# Column sums:                 "i j -> j"
# Batch matmul:                "b i j, b j k -> b i k"
# Batch dot:                   "b i, b i -> b"

# Attention scores (Q @ K^T per batch, per head):
#   queries: [batch, seq_q, d_model]
#   keys:    [batch, seq_k, d_model]
#   "b q d, b k d -> b q k"

# Apply attention to values:
#   attn:    [batch, q, k]
#   values:  [batch, k, d_model]
#   "b q k, b k d -> b q d"
```

### Learning
- An axis appearing only in the **output** is invalid for einsum (it can't be made up from nothing). Use `einops.repeat` if you need to broadcast a new axis.
- For matmul-shaped ops, the shared axis is the "contracted" one and disappears from the output. This is the same intuition as `i j, j k -> i k`: `j` is shared, gets summed, only `i` and `k` survive.

---

## Cross-cutting takeaways

1. **Default to einops over `.view` / `.permute`** for anything beyond trivial reshapes. The axis labels make the intent explicit and the bugs visible.
2. **`keepdim=True` on reductions** when you plan to divide or subtract; saves you the broadcasting headache.
3. **Subtract the max before exp** in any softmax/logsumexp pipeline. Always.
4. **`tensor.max(dim=...).values`** to get the values out of the namedtuple.
5. **`gather` for per-row lookups, integer indexing otherwise.** And `matrix[tuple(coords.T)]` for arbitrary-rank coordinate lookups.
6. **`(a b)` vs `(b a)` in einops** is outer-vs-inner ordering. Get this right or your tensors are silently wrong.
