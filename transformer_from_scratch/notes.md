# Transformers from Scratch - Notes

Notes from ARENA [1.1] Transformers from Scratch. Companion to [1_1_Transformer_from_Scratch_exercises.ipynb](1_1_Transformer_from_Scratch_exercises.ipynb).

A first-principles GPT-2 reimplementation, using the TransformerLens parameter conventions. The point is not to write something faster than HuggingFace, but to make every shape, weight, and activation legible enough that you can later do mech-interp on a real model and trust your mental picture.

Reference model: `gpt2-small` via `HookedTransformer.from_pretrained(...)`. The custom `DemoTransformer` is built to be `load_state_dict`-compatible with it.

---

## Big picture (what a transformer *is*, operationally)

- Input: token IDs of shape `(batch, seq)`. Each ID is an integer in `[0, d_vocab)`.
- Output: logits of shape `(batch, seq, d_vocab)`. `logits[b, j, :]` is the model's distribution over the `(j+1)`-th token in sequence `b`. **Every position predicts the next token, in parallel.**
- Inside: a **residual stream** of shape `(batch, seq, d_model)` flows top to bottom. Each layer **reads** from it via LayerNorm, computes something, and **adds** the result back. Nothing else writes to the residual stream.
- Two kinds of layer in a block:
  - **Attention** moves information *across positions* (the only thing that does this), but operates *independently per head*.
  - **MLP** processes each position *independently* (no token mixing). 4x hidden expansion + GELU + projection.
- Output: LayerNorm on the final residual stream, then `W_U` (unembed) to logits. Softmax + sample (or argmax) gives a token.

The thing that took longest to internalize: **attention is the only inter-token operation**. MLPs are pointwise.

---

## GPT-2 config (from the notebook)

```python
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072         # = 4 * d_model
    n_heads: int = 12
    n_layers: int = 12        # number of blocks; total "layers" is 2x this
```

`d_model = n_heads * d_head` and `d_mlp = 4 * d_model` are convention, not enforced.

---

## 1. Understanding inputs & outputs

### Tasks
- Loaded `reference_gpt2` via `HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)`. The three `False` flags keep the parameters in their raw form so the custom implementation can match them exactly.
- Walked through the generation loop manually:
  1. `to_tokens(text)` -> `(1, seq)` token IDs (prepends BOS by default).
  2. `model(tokens)` -> `(1, seq, d_vocab)` logits.
  3. `logits.softmax(dim=-1)` -> probabilities (same shape).
  4. `logits[0, -1].argmax(-1)` -> single next-token ID (predict from the last position only at generation time).
  5. Concatenate, re-run the model on the longer sequence.
- Examined the tokenizer: BPE, case-sensitive (`"Ralph"`, `" Ralph"`, `" ralph"`, `"ralph"` are four different tokens), arithmetic tokenizes inconsistently.

```python
# Minimal greedy generation, recomputing the whole sequence each step.
tokens = reference_gpt2.to_tokens(reference_text).to(device)
for _ in range(10):
    logits = reference_gpt2(tokens)              # (1, seq, d_vocab)
    next_token = logits[0, -1].argmax(dim=-1)    # scalar
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
```

### Learning
- Logits at position `j` are about token `j+1`. This off-by-one shows up everywhere (loss shifts by one, eval shifts by one).
- **Tokenization is leaky.** Leading space is part of the token. Common phrases form single tokens. Numbers split weirdly. Models that look bad at arithmetic are partly being failed by their tokenizer.
- `from_pretrained(..., fold_ln=False, center_unembed=False, center_writing_weights=False)` is what you want when reimplementing the model. The defaults rewrite weights into a numerically equivalent form that won't match your fresh implementation.

---

## 2. Clean transformer implementation

The implementation is module-by-module: `LayerNorm`, `Embed`, `PosEmbed`, `Attention`, `MLP`, `TransformerBlock`, `Unembed`, then assemble in `DemoTransformer`. Every module is tested by `rand_float_test` / `rand_int_test` (shape sanity) and `load_gpt2_test` (numerical match against the reference model on the same input).

### Tasks
- **LayerNorm**: mean-zero, variance-1 along the last dim, then learnable per-feature scale and shift. Used at the start of each sublayer (pre-norm), so attention/MLP read a normalized residual stream but write to the un-normalized one.
- **Embed**: token IDs -> `d_model` vectors via `W_E[tokens]`. A plain lookup. Initialized normal with `std=0.02`.
- **PosEmbed**: learned absolute positional embeddings. `W_pos` has shape `(n_ctx, d_model)`; for an input of seq length `s`, take `W_pos[:s]` and broadcast across the batch. Added to the token embedding.
- **Attention (causal multi-head)**:
  - Per head: `W_Q, W_K, W_V` are `(n_heads, d_model, d_head)`, biases `(n_heads, d_head)`.
  - Output projection `W_O` is `(n_heads, d_head, d_model)`, bias `(d_model,)`.
  - `q, k, v`: `(batch, pos, head, d_head)`.
  - Scores: `q @ k^T` summed over `d_head` -> `(batch, head, pos_Q, pos_K)`.
  - Scale by `1/sqrt(d_head)`, mask out future positions with `-inf`, softmax over key positions.
  - Weighted sum of values, then concat heads via the head-dim-mixing `W_O`.
- **`apply_causal_mask`**: build `triu(ones, diagonal=1)` over `(query_pos, key_pos)`, `masked_fill_` with `-inf` (registered as `IGNORE` buffer so device follows the module).
- **MLP**: `Linear(d_model, d_mlp) + GELU + Linear(d_mlp, d_model)`. Uses `gelu_new`.
- **TransformerBlock**: pre-norm + residual. `x = x + attn(ln1(x))` then `x = x + mlp(ln2(x))`.
- **Unembed**: linear `(d_model, d_vocab)`. Bias is registered as a parameter but with `requires_grad=False` (matches GPT-2).
- **DemoTransformer**: stitch embed + pos_embed + n_layers blocks + ln_final + unembed.

```python
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual):
        mean = residual.mean(dim=-1, keepdim=True)
        # unbiased=False: divide by N, not N-1. GPT-2 matches this.
        var = residual.var(dim=-1, keepdim=True, unbiased=False)
        # Important: sqrt(var + eps), NOT std + eps.
        normalized = (residual - mean) / t.sqrt(var + self.cfg.layer_norm_eps)
        return normalized * self.w + self.b


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=cfg.init_range)

    def forward(self, tokens):
        # tokens: (batch, pos) int. Lookup, not matmul.
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=cfg.init_range)

    def forward(self, tokens):
        batch, seq_len = tokens.shape
        # W_pos has n_ctx slots; only use the first seq_len, broadcast over batch.
        return einops.repeat(self.W_pos[:seq_len], "s d -> b s d", b=batch)


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model,)))
        for W in (self.W_Q, self.W_K, self.W_V, self.W_O):
            nn.init.normal_(W, std=cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def apply_causal_mask(self, attn_scores):
        q_pos, k_pos = attn_scores.shape[-2], attn_scores.shape[-1]
        # triu with diagonal=1 keeps only strictly-above-diagonal = future positions.
        mask = t.triu(t.ones(q_pos, k_pos, device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

    def forward(self, normalized_resid_pre):
        # q, k, v: (batch, pos, head, d_head). Keep head before pos for the bias broadcast.
        q = einops.einsum(normalized_resid_pre, self.W_Q,
                          "b pos d_m, h d_m d_h -> b pos h d_h") + self.b_Q
        k = einops.einsum(normalized_resid_pre, self.W_K,
                          "b pos d_m, h d_m d_h -> b pos h d_h") + self.b_K
        v = einops.einsum(normalized_resid_pre, self.W_V,
                          "b pos d_m, h d_m d_h -> b pos h d_h") + self.b_V

        # Dot-product scores: (b, head, pos_Q, pos_K). Sum is over d_head.
        attn_scores = einops.einsum(q, k,
                                    "b qpos h d_h, b kpos h d_h -> b h qpos kpos")
        attn_scores = attn_scores / (self.cfg.d_head ** 0.5)
        attn_scores = self.apply_causal_mask(attn_scores)
        # Softmax over key positions (the axis we're mixing along).
        attn_probs = attn_scores.softmax(dim=-1)

        # Weighted value sum, back to (b, pos, head, d_head).
        z = einops.einsum(attn_probs, v,
                          "b h qpos kpos, b kpos h d_h -> b qpos h d_h")

        # Mix heads back into d_model.
        attn_out = einops.einsum(z, self.W_O,
                                 "b qpos h d_h, h d_h d_m -> b qpos d_m") + self.b_O
        return attn_out


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp,)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model,)))
        for W in (self.W_in, self.W_out):
            nn.init.normal_(W, std=cfg.init_range)

    def forward(self, normalized_resid_mid):
        up = einops.einsum(normalized_resid_mid, self.W_in,
                           "b pos d_m, d_m d_mlp -> b pos d_mlp") + self.b_in
        act = gelu_new(up)
        down = einops.einsum(act, self.W_out,
                             "b pos d_mlp, d_mlp d_m -> b pos d_m") + self.b_out
        return down


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre):
        # Pre-norm: normalize for the sublayer, but write the sublayer output
        # to the un-normalized residual stream.
        resid_mid = resid_pre + self.attn(self.ln1(resid_pre))
        resid_post = resid_mid + self.mlp(self.ln2(resid_mid))
        return resid_post


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=cfg.init_range)
        # GPT-2 has b_U but it's not learned (kept at zero).
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab,), requires_grad=False))

    def forward(self, normalized_resid_final):
        return einops.einsum(normalized_resid_final, self.W_U,
                             "b pos d_m, d_m d_v -> b pos d_v")


class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        res = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            res = block(res)
        return self.unembed(self.ln_final(res))
```

### Results
- `load_state_dict(reference_gpt2.state_dict(), strict=False)` loads cleanly into `DemoTransformer` and reproduces the reference logits to within float precision. Numerical match on every sublayer's `load_gpt2_test` confirms the parameter layout is right.
- The Q/K/V shape convention `(batch, pos, head, d_head)` (head *after* pos) is what makes the per-head biases `(head, d_head)` broadcast correctly. Swapping to `(batch, head, pos, d_head)` before adding biases breaks the broadcast.

### Learning
- **LayerNorm: `sqrt(var + eps)`, not `std + eps`.** The two look similar and differ by ~1% for typical activations, which is exactly the kind of bug that lets every test pass except `TransformerBlock` (which compounds the error).
- **Pre-norm vs post-norm matters.** In pre-norm (what GPT-2 uses) the residual stream stays "raw"; LayerNorm only fronts the sublayer inputs. This is what makes the residual stream meaningful as an interpretability object: the "real signal" persists across layers.
- **Causal mask = `triu(diagonal=1)`.** Strictly above the diagonal is "future". `diagonal=0` would mask the diagonal too and break self-attention.
- The `IGNORE` constant lives as a registered buffer so it follows the module to the right device/dtype. Don't hard-code `float("-inf")` inside `forward`; you'll get device mismatches.
- Embedding is a lookup (`W_E[tokens]`), not a matmul. Conceptually it's `one_hot(tokens) @ W_E`, but you never materialize the one-hot.

---

## 3. Training a transformer

A "tiny GPT-2" (`d_model=32, n_heads=16, d_head=2, n_layers=4, n_ctx=128`, full GPT-2 vocab) trained on TinyStories from scratch. The point is the loop, not the result.

### Tasks
- Built `model_cfg` and instantiated a fresh `DemoTransformer`.
- Loaded TinyStories, used `tokenize_and_concatenate` with `add_bos_token=True` to produce a single packed stream of token IDs, then `train_test_split(test_size=1000)` and wrapped in `DataLoader`s.
- Wrote `get_log_probs(logits, tokens)`: shifts logits by one and `gather`s the log-prob of the actual next token. The off-by-one is the entire point.
- Wrote `TransformerTrainer` with AdamW + weight decay, wandb logging, `training_step`, `evaluate` (next-token accuracy on the test set), and a `train` loop that samples greedy completions at the end of each epoch.

```python
def get_log_probs(logits, tokens):
    """Cross-entropy in disguise. Returns log P(tokens[:, t+1] | tokens[:, :t+1])."""
    log_probs = logits.log_softmax(dim=-1)
    # Shift by one: skip the last logit (no next token to compare to) and the first token (no previous logit).
    return log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)


class TransformerTrainer:
    def __init__(self, args, model):
        self.model = model
        self.args = args
        self.optimizer = t.optim.AdamW(model.parameters(),
                                       lr=args.lr, weight_decay=args.weight_decay)
        self.step = 0
        # DataLoaders omitted for brevity.

    def training_step(self, batch):
        tokens = batch["tokens"].to(device)
        logits = self.model(tokens)
        loss = -get_log_probs(logits, tokens).mean()  # NLL == cross-entropy here
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        wandb.log({"train_loss": loss}, step=self.step)
        return loss

    @t.inference_mode()
    def evaluate(self):
        self.model.eval()
        total_correct, total_samples = 0, 0
        for batch in self.test_loader:
            tokens = batch["tokens"].to(device)
            # Drop the last position (no next token to score against).
            logits = self.model(tokens)[:, :-1]
            preds = logits.argmax(dim=-1)
            total_correct += (preds == tokens[:, 1:]).sum().item()
            total_samples += tokens.size(0) * (tokens.size(1) - 1)
        self.model.train()
        return total_correct / total_samples
```

### Results
- Loss curve: starts near `log(d_vocab) ≈ 10.8` (uniform-distribution baseline), drops sharply, then plateaus.
- The plateau corresponds roughly to learning **unigram + bigram** statistics. The "entropy of training data" baseline (`-sum p_x log p_x` over the training-token distribution) is the unigram lower bound; bigram statistics get you somewhat below that. Going further requires structure beyond local n-grams, which is what the rest of training works toward.

### Learning
- The off-by-one in `get_log_probs` (`logits[:, :-1]` paired with `tokens[:, 1:]`) is the single most error-prone line in the whole notebook. Get it wrong and your loss is computed against the *current* token, which the model can trivially achieve by being an identity function; loss will collapse and the model will learn nothing.
- The shape of the loss curve is a *diagnostic*: a fast initial drop is the model learning the unigram distribution, the second knee is bigrams, then it plateaus until something architectural lets it learn longer-range structure.
- Use `wandb.log({...}, step=self.step)` consistently. Mixing implicit step counters with explicit ones gives plots that look fine but are off by one batch.

---

## 4. Sampling from a transformer

Once trained, sampling decides *how* you turn a probability distribution over tokens into actual generated text. The notebook implements: greedy, basic categorical, temperature, frequency penalty, top-k, top-p (nucleus), and beam search. All live as `@staticmethod`s on `TransformerSampler`, dispatched by `sample_next_token` according to which kwargs are set.

### Tasks
- `sample`: autoregressive loop. Encode prompt, repeatedly compute logits, take `logits[0, -1]`, call `sample_next_token(...)`, append, stop on `max_tokens_generated` or EOS.
- `sample_next_token`: dispatcher. Order matters: temperature -> frequency penalty -> (top-k XOR top-p XOR basic). `temperature == 0` shortcuts to greedy. Top-k and top-p are mutually exclusive.
- `greedy_search`: `argmax(logits)`.
- `sample_basic`: `Categorical(logits=logits).sample()`. Use logits, not probs, so unnormalized is fine.
- `apply_temperature`: `logits / temperature`. Lower temp = sharper, higher temp = flatter. Temperature is the most undersold knob in sampling.
- `apply_frequency_penalty`: `logits - freq_penalty * bincount(input_ids, minlength=d_vocab)`. Needs `minlength=d_vocab` so the bincount tensor matches logits.
- `sample_top_k`: `logits.topk(k)` -> sample within those k by their (now relative) logits. Stay in log-space.
- `sample_top_p` (nucleus): sort logits descending, cumulative-sum the softmax probs, keep the smallest prefix whose cumulative prob is `>= top_p`, set the rest to `-inf`, sample.
- `Beams` class and `beam_search`: maintain `num_beams` partial completions with their cumulative log-prob. Each step: `generate` expands every beam by its top `k` next tokens, then `filter` keeps the best `num_beams` of the (`num_beams * k`) candidates. Optional `no_repeat_ngram_size` bans next-tokens that would close a previously-seen n-gram.

```python
class TransformerSampler:
    def __init__(self, model, tokenizer):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt, max_tokens_generated=100, verbose=False, **kwargs):
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        for _ in range(max_tokens_generated):
            logits = self.model(input_ids)
            next_token_id = TransformerSampler.sample_next_token(
                input_ids[0], logits[0, -1], **kwargs)
            input_ids = t.cat([input_ids, t.tensor([[next_token_id]], device=device)], dim=-1)
            if next_token_id == getattr(self.tokenizer, "eos_token_id", None):
                break
        return self.tokenizer.decode(input_ids[0])

    @staticmethod
    def sample_next_token(input_ids, logits, temperature=1.0, top_k=0, top_p=0.0,
                          frequency_penalty=0.0, seed=None):
        assert 0 <= top_p <= 1.0 and 0 <= top_k
        assert not (top_p != 0 and top_k != 0), "top-p and top-k are mutually exclusive"
        if seed is not None:
            t.manual_seed(seed); np.random.seed(seed)
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        if temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(input_ids, logits, frequency_penalty)
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits):
        return t.argmax(logits).item()

    @staticmethod
    def sample_basic(logits):
        # Categorical accepts unnormalized logits, no need to softmax first.
        return t.distributions.Categorical(logits=logits).sample().item()

    @staticmethod
    def apply_temperature(logits, temperature):
        return logits / temperature

    @staticmethod
    def apply_frequency_penalty(input_ids, logits, freq_penalty):
        d_vocab = logits.size(0)
        # minlength is required so the count vector aligns with logits shape.
        id_freqs = t.bincount(input_ids, minlength=d_vocab)
        return logits - freq_penalty * id_freqs

    @staticmethod
    def sample_top_k(logits, k):
        top_k_logits, top_k_token_ids = logits.topk(k)
        idx = t.distributions.Categorical(logits=top_k_logits).sample()
        return top_k_token_ids[idx].item()

    @staticmethod
    def sample_top_p(logits, top_p, min_tokens_to_keep=1):
        sorted_logits, sorted_idx = logits.sort(descending=True)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Number of tokens to keep is the first prefix whose cumulative prob >= top_p.
        n_keep = max(min_tokens_to_keep, int((cum_probs < top_p).sum().item()) + 1)
        kept_logits = sorted_logits[:n_keep]
        kept_idx = sorted_idx[:n_keep]
        sampled = t.distributions.Categorical(logits=kept_logits).sample()
        return kept_idx[sampled].item()
```

Beam-search core (full class in the notebook):

```python
@dataclass
class Beams:
    model: DemoTransformer
    tokenizer: GPT2TokenizerFast
    logprob_sums: Float[Tensor, " batch"]
    tokens: Int[Tensor, "batch seq"]

    def generate(self, k, no_repeat_ngram_size=None):
        """Expand each beam by its top-k continuations -> (batch * k) new beams."""
        logits = self.model(self.tokens)
        next_logprobs = logits[:, -1, :].log_softmax(dim=-1)   # (batch, d_vocab)

        if no_repeat_ngram_size is not None:
            # Ban any next-token that would close an n-gram already seen in this beam.
            n = no_repeat_ngram_size
            for b in range(self.tokens.shape[0]):
                tokens = self.tokens[b].tolist()
                if len(tokens) >= n - 1:
                    prefix = tokens[-(n - 1):] if n > 1 else []
                    banned = [ngram[-1] for i in range(len(tokens) - n + 1)
                              if (ngram := tokens[i:i + n])[:-1] == prefix]
                    if banned:
                        next_logprobs[b, banned] = -float("inf")

        topk_logprobs, topk_tokens = next_logprobs.topk(k, dim=-1)
        new_logprob_sums = (self.logprob_sums[:, None] + topk_logprobs).flatten()
        repeated = einops.repeat(self.tokens, "b s -> b k s", k=k)
        new_tokens = t.cat([repeated, topk_tokens.unsqueeze(-1)], dim=-1)
        new_tokens = einops.rearrange(new_tokens, "b k s -> (b k) s")
        return Beams(self.model, self.tokenizer, new_logprob_sums, new_tokens)

    def filter(self, num_beams):
        """Keep the best num_beams active beams, return them and any terminated ones."""
        # Split into terminated (ends with EOS) and active; keep top num_beams of active by logprob_sum.
        ...  # see notebook
```

### Results
- **Greedy** locks the model into the highest-probability path; for trained-from-scratch tiny models this often degenerates into repetition.
- **Temperature** trades coherence for diversity. `T = 0.7` is the workhorse default; `T -> 0` approaches greedy, `T >> 1` approaches uniform.
- **Frequency penalty** noticeably tames "Baby, baby, baby..."-style loops without destroying coherence at moderate values.
- **Top-k** with `k=40` and `T=0.7` produces the most "GPT-2-paper-like" outputs - matches the unicorn-prompt setup.
- **Top-p** with `p=0.95` adapts the candidate set per-step (small set when the model is confident, larger when it's uncertain). Generally produces more natural text than fixed top-k.
- **Beam search** with `no_repeat_ngram_size=2` and `num_beams=40` finds high-probability completions that greedy can miss, at the cost of higher per-step compute and a tendency toward generic, "averaged" continuations. Better for tasks with a clearer target (translation, summarization) than open-ended generation.

### Learning
- `Categorical(logits=...)` is the right primitive. It accepts unnormalized logits, so you don't need to softmax-then-sample (which is the same thing but does the arithmetic twice).
- Top-p needs to keep at least one token even when no prefix's cumulative prob exceeds `top_p` (rare with float, but always handle it via `min_tokens_to_keep=1`).
- Beam search is *not* sampling. It's a search over likely sequences. If you want diversity, you want sampling; if you want a single high-probability completion, you want beam search.
- The `no_repeat_ngram_size` trick is a cheap and effective de-loop hack. It encodes "if the last `n-1` tokens match the start of any previously-seen n-gram, ban its completion."
- **KV caching** (bonus exercise in section 4 of the notebook): at generation time, attention's keys and values for past tokens never change. Storing them avoids recomputing the full attention each step. The speedup is real but usually smaller than the `seq_len` factor you'd naively expect, because the per-step cost is dominated by other things (MLPs, memory bandwidth).

---

## Takeaways

1. **The residual stream is the actor.** Everything else is a side function that reads it (after LayerNorm) and writes back into it. This is what makes layers composable and analyzable.
2. **Attention is the only inter-token operation.** MLPs are pointwise. If you want to know how information flows across positions, look at attention.
3. **Heads operate independently.** Same parameters, but disjoint subspaces of the residual stream. `n_heads * d_head` partitions `d_model` (sort of - actually each head reads/writes the full `d_model` via its own projections).
4. **Pre-norm vs post-norm matters for interpretability,** not just training stability. Pre-norm keeps the residual stream "raw"; pretrained pre-norm models are far easier to reason about.
5. **Tokenization is a leaky abstraction.** Numbers, casing, leading spaces, multi-byte unicode all do weird things. When the model looks dumb, blame the tokenizer first.
6. **Off-by-one in loss / eval / sampling is the universal bug.** `logits[:, :-1]` pairs with `tokens[:, 1:]`. Always.
7. **Loss-curve shape is diagnostic.** Initial drop = unigram. Second knee = bigram. Plateau = waiting on structure.
8. **Sampling is orthogonal to the model.** Same weights, different `sample_*` produces wildly different text. Tune sampling separately from training.

---

## Mini-glossary

- **Residual stream:** the running sum of all sublayer outputs, shape `(batch, seq, d_model)`. The input to every layer.
- **Pre-norm:** apply LayerNorm to the input of each sublayer, but write the sublayer's output to the un-normalized stream. GPT-2 uses this.
- **Causal mask:** `triu(ones, diagonal=1)` set to `-inf` in attention scores. Forces token `j` to only attend to positions `<= j`.
- **`d_model`, `d_head`, `d_mlp`:** residual width, per-head attention width, MLP hidden width. Usually `d_model = n_heads * d_head` and `d_mlp = 4 * d_model`.
- **`fold_ln` (in TransformerLens):** rewrites the model so LayerNorm weights are folded into adjacent linears. Numerically equivalent but breaks parameter-level comparison with a fresh implementation. Set to `False` when reimplementing.
- **Cross-entropy loss:** `-mean(log P(true_next_token))`. For language modeling, the "uniform baseline" is `log(d_vocab)`.
- **Top-k vs top-p:** fixed-size vs probability-mass-defined truncation of the next-token distribution before sampling.
- **Beam search:** maintain `num_beams` highest-log-prob partial sequences, expand each by top-`k`, keep best. Search, not sampling.
- **KV cache:** at generation time, store past attention keys/values so each new step only computes attention for the new token's query against all cached keys/values.
