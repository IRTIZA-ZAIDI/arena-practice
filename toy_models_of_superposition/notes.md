# Toy Models of Superposition & SAEs - Notes

Notes from ARENA [1.5.4] Toy Models of Superposition & Sparse Autoencoders. Companion to [1_5_4_Toy_Models_of_Superposition_&_SAEs_exercises.ipynb](1_5_4_Toy_Models_of_Superposition_&_SAEs_exercises.ipynb).

The central question: when a neural network has FEWER dimensions than features it wants to represent, how does it pack them in? And once features are packed in, how do we UNPACK them with a sparse autoencoder? The exercises walk through the Anthropic [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) paper, then extend to sparse-autoencoder recovery (standard, Gated, JumpReLU).

For a TransformerLens API cheat sheet used later, see [../indirect_object_identification/transformerlens.md](../indirect_object_identification/transformerlens.md).

---

## Big picture (what superposition actually is)

**Superposition** = a network representing more features than it has dimensions, by encoding each feature in a NON-orthogonal direction. Interference between features is tolerable if the features are SPARSE (rarely both active on the same input).

Two flavors:

1. **Bottleneck superposition** (representation).  Compress `n_features` into `d_hidden < n_features` and try to reconstruct. Studied via `h = Wx, x' = ReLU(W^T h + b)`. Non-privileged basis: any orthogonal rotation of `W` gives the same behavior.
2. **Neuron superposition** (computation).  Same setup but with a ReLU on the hidden layer: `h = ReLU(Wx)`. Now the basis IS privileged (ReLU is not rotation-invariant), and features can share neurons.

**Polysemanticity** = one neuron activating for many unrelated concepts, i.e. many features share the same neuron. This is superposition observed from the neuron side.

**Sparse Autoencoders** try to UNPACK the superposition by learning an overcomplete dictionary `z = ReLU(W_enc h + b_enc)` where `d_sae >> d_hidden` and `z` is sparse. Each active `z[i]` ideally corresponds to one interpretable feature.

The 5 exercise sections:

1. Toy model with `h = Wx`, `x' = ReLU(W^T h + b)`. Sweep over sparsity + importance + correlation. See features get packed into pentagons, antipodal pairs, etc.
2. Same model but with `h = ReLU(Wx)` (privileged basis). Add a nonlinear task: compute `abs(x)` for `x in [-1, 1]`. See asymmetric-superposition motif.
3. Feature geometry: dimensionality metric `D_i = ||W_i||^2 / sum_j(W_i_hat . W_j)^2`. Sticky points at `D = 1` and `D = 1/2`. Antipodal pairs, digons, triangles, pentagons.
4. Deep double descent: fix a small batch and train repeatedly. Model transitions from memorising (data in superposition) to generalising (features in superposition), with a test-loss spike in between.
5. SAEs on the trained toy model. Standard L1-sparse SAE, then Gated SAE (decouples gate from magnitude), then JumpReLU SAE (threshold + straight-through-estimator).

---

## 1. Toy Model: superposition in a non-privileged basis

### Tasks
- Built `ToyModel(nn.Module)` with weights `W: (n_inst, d_hidden, n_features)` and `b_final: (n_inst, n_features)`. The `n_inst` dimension lets me train many independent models in parallel (batched across instances, same optimizer).
- Implemented `forward(features)` = `ReLU(W^T (W x) + b_final)`.
- Implemented `generate_batch(batch_size)` = uniform features in `[0, 1]`, present per-position with `feature_probability` (Bernoulli).
- Implemented `calculate_loss` with importance-weighted MSE, means over batch and features, sum over instances.
- Trained sweeps: constant importance + varying sparsity, varying importance + constant sparsity.
- Visualized `W` in 2D (each column is a 2D vector), `W^T W` heatmap (5x5 pairwise cosine similarities).
- Extended to correlated / anticorrelated feature pairs and observed how the geometry shifts.

### The setup (why W^T W)

The model is:
```
h  = W x                              # (d_hidden,)  from (n_features,)
x' = ReLU(W^T h + b)                  # (n_features,)
```

`W` is shape `(d_hidden, n_features)` = `(2, 5)` in the standard demo. Effective forward pass (before ReLU) is `x -> W^T W x + b`. `W^T W` is `(n_features, n_features)` and its entries are dot products between pairs of 2D column vectors. Reading `W^T W` shows:
- Diagonal entries: squared norms of each feature's 2D embedding (approx 1 if represented, 0 if not).
- Off-diagonal entries: cosine similarities between pairs (INTERFERENCE terms).

If the diagonal is roughly 1 and off-diagonals near 0, features are cleanly separated. If some diagonals are 0, those features aren't represented. If off-diagonals are large-negative, features are packed as antipodal pairs.

### Loss
```
L = (1 / (B * F)) sum_x sum_i I_i (x_i - x_i')^2
```

- `B` = batch size, `F` = n_features.
- `I_i` = importance of feature i (weight in MSE).
- `sum_x` is over batch elements; `sum_i` is over features.
- We take the MEAN over batch and features (standard MSE convention).
- We SUM over instances (so each instance is trained at the "same rate" as a single instance).

```python
def calculate_loss(self, out, batch):
    err = self.importance * (batch - out) ** 2         # weight per feature per example
    loss = einops.reduce(err, "batch inst feats -> inst", "mean")
    return loss.sum()
```

### Sparsity + importance sweep

Set `n_features=5, d_hidden=2, n_inst=8` and sweep `feature_probability` from 1 (dense) down to 0.05 (very sparse).

- **Dense (`p ~ 1`)**: the model can only represent 2 features well (there are only 2 hidden dimensions). Features with lower importance get compressed to zero. `W` picks out the 2 most important directions.
- **Sparse (`p ~ 0.05`)**: features rarely co-occur, so INTERFERENCE is cheap. The model packs all 5 features into the 2D space at 72-degree angles (pentagon). Every feature has a nonzero norm.
- **Intermediate**: partial superposition, sometimes with a subset in antipodal pairs.

Key intuition: **superposition is a bet that features won't be active together**. With p=0.5 (dense), two features being active simultaneously happens 25% of the time, and cross-talk between them hurts the loss. With p=0.05, both being active happens 0.25% of the time - interference is worth it.

### Visualizations across sparsity

For `n_features=100, d_hidden=20` with importance decreasing as `100^-linspace(0,1,100)`:
- Each 2D subplot shows one instance's `W` columns as vectors.
- More important features get longer vectors (norm approx 1); less important features get near-zero vectors.
- Sparser instances pack more vectors into the 20D space.
- The Frobenius norm `||W||_F^2` grows with sparsity - directly measures "number of represented features".

### Correlated and anticorrelated pairs

`generate_correlated_batch` creates:
- **Correlated pair `(i, j)`**: `x_i` and `x_j` are simultaneously present or simultaneously absent. Same magnitude if present.
- **Anticorrelated pair `(i, j)`**: exactly ONE of `x_i, x_j` is present per example (mutually exclusive).

Result:
- **Correlated features** want to be REPRESENTED TOGETHER, so the model uses orthogonal directions for them (they can be simultaneously active without interference).
- **Anticorrelated features** are mutually exclusive, so the model prefers to make them ANTIPODAL (same direction, opposite sign). They never both fire so the negative-direction case never interferes with the positive-direction case.

This is a general lesson: the geometry the model learns depends heavily on the CO-OCCURRENCE STRUCTURE of the features, not just their marginal frequency.

---

## 2. Toy Model: superposition in a privileged basis

### Tasks
- Built `NeuronModel(ToyModel)` overriding `forward` with `h = ReLU(W x)`, `x' = ReLU(W^T h + b)`.
- Observed how superposition changes when the hidden layer has a preferred (neuron) basis.
- Built `NeuronComputationModel` that computes `abs(x)` instead of identity, forcing the ReLU to actually be used.
- Studied the asymmetric superposition motif in the resulting weights.

### Why add a ReLU?

Without ReLU on the hidden layer, the effective operation is linear and any orthogonal rotation of `W` yields the same model. This is a NON-privileged basis: individual "neurons" (rows of `W`) are meaningless.

Add `ReLU` on the hidden layer -> the basis becomes privileged. `ReLU` is not equivariant under rotation. Rotating `W` and unrotating output no longer gives the same model. So the ROWS of `W` (individual "neurons") are now meaningful directions, and we can ask "what does neuron k detect?".

But there's a subtlety. If the task is `identity` (reconstruct x), the model doesn't NEED the ReLU - it can shift biases so all pre-ReLU values are positive, and then ReLU acts as identity. This is why we switch to `abs(x)` in the next model:

### Computation in superposition (abs task)

Now inputs are sampled from `[-1, 1]` and the model must output `|x|`. Since `|x| = ReLU(x) + ReLU(-x)`, the ReLU CANNOT be bypassed.

`NeuronComputationModel`:
```
h  = ReLU(W1 x)        # W1 shape (d_hidden, n_features)
x' = ReLU(W2 h + b)    # W2 shape (n_features, d_hidden)
```

Now `W1` and `W2` are INDEPENDENT (unlike the tied-weights setup of section 1).

Result: at low sparsity, each feature `i` gets TWO neurons: one for `ReLU(x_i)` and one for `ReLU(-x_i)`, with matching output weights. As sparsity increases, neurons start being shared across features (multiple features colocated in one neuron).

### Asymmetric superposition motif

At intermediate sparsity, sometimes one neuron detects both feature `i` and feature `j`, but with DIFFERENT input weights:
- W1[k, i] = large positive
- W1[k, j] = small positive
- W2[i, k] = small positive
- W2[j, k] = large positive

When only `i` is active: output for i = `large * small` = correct, output for j = `small * small` = near zero. Good.

When only `j` is active: output for i = `large * large` = way too large. Output for j = `small * large` = correct. BAD - i gets over-activated.

The model FIXES this by using ANOTHER neuron with opposite sign, so the excess-i activation gets cancelled. This works because there's a final ReLU: as long as the corrective neuron pushes i's total NEGATIVE, ReLU truncates to zero.

Key lesson: **the model is willing to add complexity to superpose two features into one neuron, if the sparsity structure is right**. The correction motif is a real thing you can see in the visualizations.

---

## 3. Feature Geometry

### Tasks
- Computed the **squared Frobenius norm** `||W||_F^2 = sum_ij W_ij^2` as a proxy for "number of features represented".
- Computed the **dimensionality per feature** `D_i = ||W_i||^2 / sum_j(W_i_hat . W_j)^2`.
- Plotted `D_i` vs sparsity to reveal sticky geometric configurations.

### Why Frobenius norm counts features

If each feature is represented with a unit-norm 2D vector, `||W||_F^2` = sum of squared norms of columns = number of unit vectors = number of represented features (approximately).

```python
frobenius_sq = W.norm(dim=1).pow(2).sum(dim=-1)   # per instance
```

Plotting `frobenius_sq` vs sparsity shows the total representational capacity growing as sparsity increases.

### Dimensionality per feature

```
D_i = ||W_i||^2 / sum_j (W_i_hat . W_j)^2
```

where `W_i` is the 2D column vector for feature i, and `W_i_hat` is its unit-length version.

Properties:
- `D_i >= 0`, with equality iff feature is not represented.
- `D_i <= 1`, with equality iff `W_i` is orthogonal to all `W_j` for `j != i` (i.e. feature has its own dimension).
- If `k` features are all parallel and orthogonal to everything else, they share equally: `D_i = 1/k`.
- Sum of `D_i` over all i cannot exceed `d_hidden`.

```python
@t.inference_mode()
def compute_dimensionality(W):
    W_norms = W.norm(dim=1, keepdim=True)                    # (n_inst, 1, n_features)
    numerator = W_norms.squeeze() ** 2                        # (n_inst, n_features)
    W_normalized = W / (W_norms + 1e-8)
    denominator = einops.einsum(
        W_normalized, W,
        "i h f1, i h f2 -> i f1 f2",
    ).pow(2).sum(-1)                                          # (n_inst, n_features)
    return numerator / denominator
```

### The sticky points

Plotting `D_i` vs sparsity for many features + many instances reveals CONCENTRATIONS at specific fractions:

- **1**: feature has its own dimension.
- **1/2**: antipodal pair - two features share a dimension by pointing in exactly opposite directions.
- **2/5, 2/3, 3/4, ...**: various regular geometric configurations (pentagons, tetrahedra, square antiprisms, etc.)

The 1/2 sticky point in particular corresponds to the "digon" geometry: two features with `W_i = -W_j`. This works especially well for ANTICORRELATED features because they never both fire.

Between the sticky points, the model transitions via phase changes - increasing sparsity smoothly, we see abrupt reorganizations of feature geometry.

---

## 4. Superposition & Deep Double Descent

### Tasks
- Trained `ToyModel` in the double-descent regime: fixed a batch of size `B` and used the SAME batch for ALL training steps.
- Swept over batch sizes and observed train vs test loss.
- Used AdamW with weight decay to encourage weight norm to stay bounded.

### The theory

Two solutions the model can find:
1. **Memorising**: represent each DATAPOINT as a superposed direction. `B` datapoints packed into 2D. Great on train, terrible on test (unseen datapoints aren't in the memorized set).
2. **Generalising**: represent each FEATURE as a direction. Works for any datapoint. Great on both train and test.

The paper's claim: as we vary batch size:
- **Very small batch (B < n_features)**: the memorising solution is more efficient - you can pack B datapoints as antipodal or polytope arrangements. Train loss low, test loss high.
- **Medium batch (B ~ n_features)**: solutions are competing. Test loss spikes.
- **Very large batch (B >> n_features)**: memorising becomes impossible; generalising wins. Train and test loss both low.

The spike in test loss is the "double descent" - loss first goes up (as memorising becomes worse) then comes back down (as generalising takes over).

### What I ran
- 5 different batch sizes, e.g. `[3, 6, 10, 30, 100]`.
- Same weight init + same optimizer settings across sweep.
- Plotted train loss + test loss as heatmaps over training steps.
- Plotted the 2D projections of feature vectors (blue) and data (red) to see the memorising -> generalising transition visually.

### Key takeaway

Datapoints and features are BOTH candidates for what the model puts into superposition. Which wins depends on:
- How many datapoints vs features.
- Whether datapoints are effectively mutually-exclusive (like separate memorized examples) or share structure (like natural features that co-occur).

This reframes double descent: it's not a mysterious optimization glitch, it's a competition between two valid encodings of the training distribution.

---

## 5. Sparse Autoencoders in Toy Models

### Tasks
- Built `ToySAE(nn.Module)` that takes the toy model's HIDDEN state `h` (not `x`) as input.
- Implemented `__init__`, `W_dec_normalized`, `generate_batch`, `forward`, `calculate_loss`, `optimize` methods.
- Trained SAE on frozen toy-model hidden activations.
- Visualized reconstruction quality (h -> h') and latent activation patterns.
- Implemented `resample_simple` and `resample_advanced` for dead-latent recovery.
- Extended to `GatedToySAE` and `JumpReLUToySAE`.

### Problem setup

The toy model's forward pass ends with the hidden state `h` (dimension `d_in = d_hidden = 2`). We take that `h` (from many inputs) and train an SAE:

```
z  = ReLU(W_enc (h - b_dec) + b_enc)        # (d_sae,)  overcomplete + sparse
h' = W_dec z + b_dec                          # (d_in,)   reconstruction
```

Shapes (per-instance, ignoring batch):
- `h, h'` shape `(d_in,)` = `(2,)` in our small demo.
- `z` shape `(d_sae,)` = 5 or more, larger than `d_in`.
- `W_enc` shape `(d_in, d_sae)`, `W_dec` shape `(d_sae, d_in)`.
- `b_enc` shape `(d_sae,)`, `b_dec` shape `(d_in,)`.

### Why the `b_dec` centering

The forward first computes `h - b_dec`, encodes that, then reconstructs by adding `b_dec` back. Intuition: `b_dec` acts as a LEARNED CENTER for the hidden states. `W_enc` reads deviations from this typical center, and `b_dec` gets restored at the output.

This is orthogonal to sparsity - the `b_dec` step doesn't affect L1 or activation patterns, it just gives the encoder a consistent input distribution.

### Why untied weights

`W_enc` and `W_dec` are learned separately (not `W_dec = W_enc^T`). This gives the SAE more flexibility - reading a feature and writing it back are different operations, and untying them helps at high sparsity coefficients.

### Loss

```
L = ||h - h'||^2   +  lambda * ||W_dec_normalized_row_norms * z||_1
```

The reconstruction MSE plus L1 regularization on activations. The L1 is scaled by decoder row norms to prevent the model from "cheating" by scaling `z` down and `W_dec` up (a rescaling that shrinks L1 without changing the reconstruction).

### Training details
- Adam optimizer with LR schedule.
- Feed batches of `h` = generated feature vectors `x` -> `h = W x` from the frozen toy model.
- Track L0 (mean number of active latents per input), reconstruction MSE, sparsity loss separately.
- Visualize decoder columns `W_dec[i]` as 2D vectors overlaid with the toy model's feature vectors - active latents should align with real features.

### Resampling dead latents

Some SAE latents never fire during training (dead). These waste capacity.

`resample_simple`:
- Every N steps, find latents that haven't fired in the last M steps.
- Pick a random subset of DATA POINTS (`h` inputs) that the current SAE reconstructs poorly.
- Replace the encoder weights for dead latents with random normalized copies of those hard-to-reconstruct `h`.
- Zero out the decoder weights and biases for those latents (so they start fresh).

`resample_advanced` (from Anthropic's monosemanticity paper):
- Weight the "hard-to-reconstruct" sampling by reconstruction loss squared.
- After resampling, rescale the encoder/decoder weights so the resampled latents have activation magnitudes similar to alive latents.

### Metrics to watch during SAE training
- **L0** = mean number of nonzero `z` entries per token. Target: comparable to true feature sparsity.
- **Reconstruction MSE**: how well `h'` matches `h`.
- **Dead latent fraction**: what % of `z` entries never fire.
- **Feature recovery**: for each true feature, is there an SAE latent aligned with it?

---

## 6. Gated SAEs

The DeepMind [Gated SAE paper](https://arxiv.org/pdf/2404.16014) motivates the architecture from two problems with standard SAEs:

### Problem 1: features are effectively binary
Real features often are "on" or "off" (this is about basketball, or it isn't). The continuous ReLU output is a proxy - we'd like an architecture that explicitly represents the binary "is active" separately from the "how strong".

### Problem 2: shrinkage
The L1 penalty pushes ALL activations toward zero, including the ones that should be large. So even the best latent's activation is biased down, and reconstruction suffers.

### The gated architecture

Replace `f(x) = ReLU(W_enc(x - b_dec) + b_enc)` with:

```
pi_gate(x)  = W_gate(x - b_dec) + b_gate            # gate preactivation
f_gate(x)   = 1[pi_gate(x) > 0]                     # binary gate  (Heaviside)
f_mag(x)    = ReLU(W_mag(x - b_dec) + b_mag)        # magnitude
f(x)        = f_gate(x) * f_mag(x)                  # elementwise
```

Two separate paths: one decides IF the latent is active (binary), one decides HOW STRONG (continuous). Reconstruction is `x_hat = W_dec f(x) + b_dec`.

### The tied variant

`W_mag[i, j] = exp(r_mag)[i] * W_gate[i, j]`. Now `W_mag` shares direction with `W_gate` up to a per-latent scale. This turns out to be almost equivalent to a JumpReLU (see next section) with a parameterized threshold.

### Loss with auxiliary term

Can't put L1 on `f_gate` directly - the Heaviside is discontinuous, gradients don't propagate. Instead put L1 on the pre-activation:

```
L_sparsity = lambda * ||ReLU(pi_gate(x))||_1
```

But there's a residual problem: gradient can flow through `pi_gate` only via `f_mag`, and the gate could learn "always off" while `f_mag` compensates. Solution: add an AUXILIARY loss that ties the gate's ability to reconstruct:

```
L_aux = ||x - x_hat(ReLU(pi_gate(x)))||^2
```

with `x_hat` using a DETACHED `W_dec`. This forces the gate to at least be capable of reconstruction on its own, preventing degenerate gate=off solutions.

Full Gated SAE loss:

```
L = ||x - x_hat(f(x))||^2  +  lambda * ||ReLU(pi_gate(x))||_1  +  L_aux
```

### What this fixes
- Gates are binary, so at inference time an active latent is either "on" or "off" - clean interpretation.
- Magnitudes are learned by a path with NO L1 penalty, so no shrinkage.
- Gradient flow is well-behaved via the auxiliary loss.

Empirically: Gated SAEs beat standard SAEs at the reconstruction-vs-L0 Pareto frontier.

---

## 7. JumpReLU SAEs

Same motivation as Gated (binary-ish gates + no shrinkage), but simpler.

### The architecture

Identical to standard SAE except the activation function is:

```
JumpReLU_theta(z) = z * H(z - theta)
```

where `H` is the Heaviside step function and `theta` is a LEARNED per-latent threshold. So a latent's activation is zero when `z < theta`, and equal to `z` when `z >= theta`. There's a JUMP at `theta`.

Learn `theta > 0` by parameterizing `theta = exp(log_theta)` (unconstrained param).

### Loss

Direct L0 penalty (not L1):

```
L = ||x - x_hat||^2  +  lambda * ||f(x)||_0
```

where `||f(x)||_0 = sum_i H(z_i - theta_i)` counts nonzero activations.

But both Heaviside and the L0 norm are discontinuous - how do we backprop through them?

### Straight-through estimators

For a term `H(z - theta)`, we approximate its derivative wrt `theta` as:

```
d/d theta H(z - theta) := -(1/eps) K((z - theta) / eps)
```

where `K` is a KERNEL function (a probability density: non-negative, integrates to 1, centered at 0). The Anthropic/DeepMind experiments use the RECTANGLE kernel:

```
K(u) = H(u + 1/2) - H(u - 1/2)             # 1 on [-1/2, 1/2], else 0
```

Similarly for `JumpReLU`:

```
d/d theta JumpReLU_theta(z) := -(theta / eps) K((z - theta) / eps)
```

### Why this works (functional intuition)

We're replacing a discontinuous function with a SMOOTH APPROXIMATION whose derivative is a bump function centered at the jump. The kernel bandwidth `eps` sets how "spread out" the fake gradient is.

Integrating our approximated gradient over a region containing the discontinuity gives the size of the jump (=1 for H, =z for JumpReLU) - so our approximation has the correct FIRST MOMENT, which is what optimization cares about.

### Why this works (probabilistic intuition)

Our inputs `x` are random. The expected loss `E_x[L_theta(x)]` IS smooth in `theta`, even though `L_theta(x)` isn't. The kernel approximation is essentially KERNEL DENSITY ESTIMATION of the derivative of the expected loss. We estimate the smooth thing from finite-sample noisy samples.

### PyTorch implementation

Use `torch.autograd.Function` to give a custom backward for JumpReLU:

```python
class JumpReLU(t.autograd.Function):
    @staticmethod
    def forward(ctx, z, theta, eps):
        ctx.save_for_backward(z, theta)
        ctx.eps = eps
        return z * (z > theta).float()   # JumpReLU

    @staticmethod
    def backward(ctx, grad_out):
        z, theta = ctx.saved_tensors
        eps = ctx.eps
        # Rectangle kernel: nonzero when |z - theta| < eps/2
        in_kernel = (abs(z - theta) < eps / 2).float() / eps
        # z-gradient: passes through when z > theta, else 0
        grad_z     = grad_out * (z > theta).float()
        # theta-gradient via STE
        grad_theta = grad_out * (-theta) * in_kernel
        return grad_z, grad_theta, None
```

Similarly for the Heaviside function used in the L0 penalty. Both use the same rectangle kernel.

### Tradeoffs
- JumpReLU is SIMPLER than Gated (one path, no auxiliary loss).
- It can be a bit harder to train because gradient signal to `theta` is sparse (only when `z` is close to `theta`).
- Gated SAE ablations from the DeepMind paper suggest untying magnitude weights isn't necessary, suggesting JumpReLU (which effectively ties them) is often just as good.

Empirically: JumpReLU is competitive with Gated at the reconstruction-vs-L0 frontier.

---

## Takeaways (across all 5 sections)

- **Superposition is a bet on sparsity**. It's cheap to overload a dimension when features rarely co-occur. Interference is tolerated in exchange for representation.
- **Non-privileged vs privileged bases**. In `h = Wx` (no ReLU on hidden), `W` and `OW` are the same model - rows of `W` aren't meaningful. Add a ReLU and now rows ARE meaningful (neurons). This is the difference between residual-stream analysis and MLP-neuron analysis.
- **`W^T W` tells the whole story** in a non-privileged basis. Diagonal = per-feature representation strength; off-diagonal = per-pair interference.
- **Frobenius norm counts features; dimensionality per feature tells you which geometry**. The sticky points (1, 1/2, 2/5, ...) correspond to regular polytope arrangements.
- **Anticorrelated features pack better than uncorrelated ones**. If two features are never simultaneously active, they can share a dimension antipodally with zero interference.
- **Correlated features want ORTHOGONAL directions**, not the same one. They co-occur, so sharing a direction causes interference.
- **The identity task is a trap**. Reconstructing the input can be done linearly, so studying superposition needs a nonlinear task (like `abs(x)`). Otherwise the ReLU never fires and the model looks like a linear projection.
- **Asymmetric superposition is real**. Sometimes a single neuron encodes two features with UNEQUAL magnitudes and a corrective neuron fixes the leakage. Predicting this from theory is easier than reverse-engineering it from a trained model.
- **SAEs unpack superposition into a wider, sparse code**. Standard SAE = ReLU + L1. Gated SAE = decouple binary gate from continuous magnitude. JumpReLU SAE = threshold activation + L0 penalty via straight-through gradients.
- **Dead latents are wasted capacity**. Resample them by re-initializing to point at data the current SAE reconstructs poorly. This is one of the most important training tricks for SAEs.
- **Shrinkage is a real bug in vanilla SAEs**. L1 pushes ALL activations down, including the ones that should be big. Gated/JumpReLU fix this by removing L1 pressure from the magnitude path.
- **Deep double descent = memorisation-vs-generalisation as a superposition contest**. When batch size < n_features, packing datapoints wins. When batch size >> n_features, packing features wins. The spike between is the transition.
