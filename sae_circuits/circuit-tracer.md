# circuit-tracer tutorial

A tutorial on the [`circuit-tracer`](https://github.com/decoderesearch/circuit-tracer) library, which packages the attribution-graph pipeline from Anthropic's Circuit Tracing paper (Ameisen et al., 2025) into a small API. Companion to [notes.md](notes.md) for the SAE Circuits exercises - notes.md walks through the manual implementation from scratch, this file is the library reference.

If Section 3 of the notebook was "build the attribution graph from `nn.Module` + `torch.autograd`," Section 4 is "call `attribute(prompt, model)`." Same math, one line of code.

---

## What it is

A wrapper around the linearized-model + gradient-attribution pipeline. It:

1. Loads a model with a **stack of transcoders** attached at every layer.
2. Freezes attention patterns and LayerNorm scales.
3. Replaces each MLP's backward pass with the transcoder's linear skip connection.
4. Builds nodes (embeddings, latents, MLP errors, logits) and edges (gradient-based influence).
5. Prunes to a sparse graph.
6. Provides an intervention API (`feature_intervention`, `feature_intervention_generate`) that turns individual features up/down and reruns the model.

The whole pipeline described in Section 3 of the notebook is what `attribute()` does internally.

---

## Installation

Clone the repo alongside the ARENA exercises:

```bash
cd chapter1_transformer_interp/exercises
git clone https://github.com/decoderesearch/circuit-tracer.git
```

Then add it to `sys.path` before importing:

```python
import sys
from pathlib import Path

circuit_tracer_path = Path("chapter1_transformer_interp/exercises/circuit-tracer")
sys.path.insert(0, str(circuit_tracer_path))

from circuit_tracer import ReplacementModel, attribute
```

If VSCode complains about missing imports, add `"python.analysis.extraPaths": ["./chapter1_transformer_interp/exercises/circuit-tracer"]` to workspace settings.

---

## The two central objects

### `ReplacementModel`

A `HookedSAETransformer` (via TransformerLens) with GemmaScope transcoders pre-attached at every layer and the freeze / replacement machinery ready to go. Two ways to construct:

```python
from circuit_tracer import ReplacementModel

# Preferred: from a pretrained model + transcoder release.
model = ReplacementModel.from_pretrained(
    "google/gemma-2-2b",         # base model id (any Gemma with GemmaScope)
    "gemma",                     # replacement config preset ("gemma" or "gemma-3")
    dtype=torch.bfloat16,
    backend="transformerlens",   # or "hf" when TL introspection isn't needed
)
```

The two-arg call is the important one. `"gemma"` here is the **replacement config** name, not the model family - it points to a bundled config that says "use GemmaScope 2 transcoders at width=16k, l0=small, affine=True, instruction_tuned matching the base model."

Available presets (check `circuit_tracer/configs/`): `"gemma"`, `"gemma-3"`, `"gemma-3-1b"`, `"gpt2-small"`, etc. If none matches, construct the `ReplacementModel` directly by passing a `TranscoderSet` object with specific SAELens IDs.

Once loaded, `model` behaves like a regular `HookedSAETransformer` for tokenization and standard forward passes, plus a handful of circuit-tracer-specific methods.

### `attribute(prompt, model)`

The main entry point. Runs the entire Section-3 pipeline:

```python
from circuit_tracer import attribute as circuit_tracer_attribute

graph = circuit_tracer_attribute(
    prompt="Fact: the capital of the state containing Dallas is",
    model=model,
    verbose=True,               # progress bars
    max_n_logits=10,            # top-k output tokens to use as targets
    desired_logit_prob=0.95,    # or stop when cumulative logit prob crosses this
    batch_size=8,               # target-node batches for backward passes
    edge_threshold=0.98,        # keep edges up to 98% of total influence per row
    node_threshold=0.8,         # keep nodes up to 80% of total influence
    max_feature_nodes=None,     # cap on total latent nodes (None = no cap)
)
```

Returns a `Graph` object (dataclass wrapping the pruned adjacency matrix, node metadata, and top-token info). The graph is the input to visualizations and interventions.

Key behaviour to know:
- `attribute` runs its own forward + backward passes internally with all hooks installed. No manual freeze/replacement setup required.
- Pruning happens in two stages: keep enough edges per target to cover `edge_threshold` of the total row weight, then keep enough nodes overall to cover `node_threshold`. Higher thresholds -> denser graph -> more information but harder to read.
- Runtime scales with `max_n_logits * n_layers * n_positions`. Long prompts or high logit-target counts get slow.

---

## Getting activations for interventions

Separately from graph attribution, the actual sparse latent activations are often needed (to know default activation values, or to feed into cross-prompt swaps):

```python
logits, activations = model.get_activations(prompt, sparse=True)
```

`activations` is a dict-like structure indexed by `(layer, position, feature_idx)` -> activation value. Sparse form skips zero activations, which is most of them.

---

## Feature interventions

Two methods, same interface. Interventions are lists of `(layer, pos, feature_idx, value)` tuples.

### Single-step: `feature_intervention`

Runs one forward pass with the specified features clamped to the given values (overriding whatever the transcoder computed):

```python
interventions = [
    (23, 10, 12237, -2.0),   # ablate "Say Austin" at position 10 in layer 23
    (20, 10, 15589, 0.0),    # zero another feature
]

new_logits, new_activations = model.feature_intervention(prompt, interventions)
```

Returns `(logits, activations)` where activations reflect the state *after* the intervention (downstream latents will have reacted to the change).

**Convention**: `value` here is the *absolute* activation to set the feature to. To "scale current activation by -2x," compute `-2 * default_activation` first.

### Multi-token generation: `feature_intervention_generate`

Same interface but generates multiple tokens with interventions sustained across all of them:

```python
text = model.feature_intervention_generate(
    prompt,
    interventions,
    do_sample=False,          # greedy
    max_new_tokens=20,
    verbose=False,
)[0]                          # returns a list of completions
```

**Gotcha**: fixed positions don't work for multi-token generation. Writing `interventions = [(23, 10, 12237, 0.0)]` means the intervention only fires when position 10 is the last position (the initial forward). Subsequent generated tokens are at positions 11, 12, ..., and the intervention doesn't fire.

**Fix**: use a `slice(seq_len - 1, None, None)` as the position. This means "the last position, whatever it currently is." The intervention will fire at every autoregressive step.

```python
seq_len = len(model.tokenizer(prompt).input_ids)
interventions_sustained = [
    (layer, slice(seq_len - 1, None), feat_idx, value)
    for layer, pos, feat_idx, value in interventions
]

text = model.feature_intervention_generate(
    prompt, interventions_sustained, do_sample=False, max_new_tokens=20,
)[0]
```

---

## Working with supernodes

For anything beyond a handful of features, group them into named clusters (a `Supernode`) and intervene at that level.

```python
from part42_sae_circuits.utils import Feature, Supernode, InterventionGraph

# One Supernode = one conceptual role.
texas_node = Supernode(
    name="Texas",
    features=[
        Feature(layer=20, pos=9, feature_idx=15589),
        Feature(layer=20, pos=10, feature_idx=15589),
        # ... typically 5-20 features per supernode
    ],
)

say_austin_node = Supernode(name="Say Austin", features=[Feature(23, 10, 12237)])

# InterventionGraph arranges Supernodes into a display order for visualization.
graph = InterventionGraph(
    ordered_nodes=[
        [capital_emb, state_emb],           # layer 0: embeddings
        [capital_node, state_node, dallas_node],   # early features
        [say_capital_node, texas_node],     # intermediate
        [say_austin_node],                  # output
    ],
    prompt=dallas_prompt,
)

# Populate default activations for each supernode from a baseline forward.
_, dallas_acts = model.get_activations(dallas_prompt, sparse=True)
for node in [capital_node, state_node, dallas_node, texas_node, say_capital_node, say_austin_node]:
    graph.initialize_node(node, dallas_acts)
```

Then a **supernode ablation** is: for each feature in the supernode, set its activation to `scaling_factor * default_value`. `-2x` is the paper's convention for "strongly suppress."

```python
from collections import namedtuple
Intervention = namedtuple("Intervention", ["supernode", "scaling_factor"])

def supernode_intervention(model, graph, interventions):
    tuples = []
    for inv in interventions:
        for i, feat in enumerate(inv.supernode.features):
            default = inv.supernode.default_activations[i].item()
            tuples.append((*feat, inv.scaling_factor * default))
    new_logits, new_acts = model.feature_intervention(graph.prompt, tuples)
    return new_logits, new_acts


# Ablate "Say a capital" at -2x:
new_logits, _ = supernode_intervention(
    model, graph, [Intervention(say_capital_node, -2)],
)
```

### Extracting supernodes from Neuronpedia

If someone has already annotated a graph on Neuronpedia, the feature groupings can be extracted via URL:

```python
from part42_sae_circuits.utils import extract_supernode_features

url = "https://www.neuronpedia.org/gemma-2-2b/graph?slug=gemma-fact-dallas-austin&pinnedIds=..."
supernode_features = extract_supernode_features(url)
# Returns dict[supernode_name, list[Feature]] parsed from the URL's clerps + pinnedIds params.
```

---

## Cross-prompt feature swaps

Turn off features from prompt A, turn on features from prompt B at their B-activation values:

```python
def cross_prompt_swap(model, base_prompt, swap_prompt,
                      features_off, features_on, scale=2.0):
    _, swap_acts = model.get_activations(swap_prompt, sparse=True)
    interventions = [(*f, 0.0) for f in features_off]
    interventions += [(*f, scale * swap_acts[f]) for f in features_on]
    _, modified_logits = model.feature_intervention(base_prompt, [])
    _, modified_logits = model.feature_intervention(base_prompt, interventions)
    return modified_logits

# Swap Texas -> California, expect Sacramento at the output.
modified = cross_prompt_swap(
    model,
    base_prompt="Fact: the capital of the state containing Dallas is",
    swap_prompt="Fact: the capital of the state containing Oakland is",
    features_off=[(layer, pos, feat) for (layer, pos, feat) in texas_features],
    features_on=[(layer, pos, feat) for (layer, pos, feat) in california_features],
    scale=2.0,
)
```

This is stronger than ablation: it demonstrates *compositional slot-filling*. If Texas is genuinely the "state" slot in the circuit, replacing it with California should cleanly redirect the answer.

---

## Visualization

`create_attribution_dashboard(result, model)` produces an interactive HTML with nodes laid out by layer, edges weighted by influence, and per-latent activation histograms. For the histograms to populate, download Gemma Scope's pre-computed example data:

```python
from part42_sae_circuits.utils import (
    load_example_data_parallel,
    create_attribution_dashboard,
)

example_data_by_layer = load_example_data_parallel(
    layers=list(range(model.cfg.n_layers)),
    model_size="1b",
    category="transcoder_all",
    width="16k",
    l0="small",
    affine=True,
    instruction_tuned=True,
)

from IPython.display import HTML
html = create_attribution_dashboard(
    result=graph, model=model, example_data_by_layer=example_data_by_layer,
)
display(HTML(html))
```

For intervention graphs (Section 4-style diagrams of supernode ablations), use `utils.create_graph_visualization(intervention_graph, top_outputs)` - it renders an SVG showing the supernodes and their current activation fractions.

---

## Quality metrics

```python
from circuit_tracer.graph import compute_graph_scores

scores = compute_graph_scores(graph, model)
# scores.replacement_score: how well the linearized model reproduces the base model's output
# scores.completeness_score: fraction of the base model's output logit-diff captured by the graph
```

Rule of thumb: replacement scores above 0.9 mean the linearization is faithful. Completeness above 0.5 is decent for a two-hop task. Both scores degrade as pruning gets more aggressive; `edge_threshold=0.7` (aggressive) giving 0.4 completeness is a sign the pruning has gone too far.

---

## Common gotchas

- **`ReplacementModel.from_pretrained` needs the `backend` kwarg**. Default may not be TransformerLens; explicitly pass `backend="transformerlens"` to get the introspection API used everywhere else in the exercises.
- **Activations are sparse-dict-like**. `activations[(layer, pos, feat)]` returns the scalar; missing keys mean zero. Don't iterate over a dense grid; iterate over the sparse keys.
- **`feature_intervention` returns absolute values**. `-2x` means "twice the default, negated," which requires multiplying by `default_activation` first. Passing `-2.0` directly sets the activation to `-2.0` regardless of default.
- **Pruning thresholds are cumulative-fraction, not absolute**. `edge_threshold=0.98` means "keep enough top edges per row to cover 98% of total row weight," not "keep edges above 0.98."
- **Multi-token generation with fixed-position interventions is a silent bug**. It just runs baseline generation and returns. Always use a `slice` for sustained interventions.
- **`extract_supernode_features` parses Neuronpedia URLs**. The URL has to have `pinnedIds` populated (i.e. someone actually clicked "pin" on the features in the Neuronpedia UI). Empty pinnedIds -> empty supernodes.
- **`get_activations(prompt, sparse=True)` vs `feature_intervention(prompt, [])`**. Both return baseline activations, but `get_activations` is faster (no gradient machinery) and the sparse dict is what most utility functions expect.

---

## Minimum working example (Dallas / Austin, end to end)

```python
import torch as t
from circuit_tracer import ReplacementModel, attribute
from part42_sae_circuits.utils import (
    Feature, Supernode, InterventionGraph, extract_supernode_features,
)
from collections import namedtuple

# 1. Load the model.
model = ReplacementModel.from_pretrained(
    "google/gemma-2-2b", "gemma",
    dtype=t.bfloat16, backend="transformerlens",
)

# 2. Run attribution.
prompt = "Fact: the capital of the state containing Dallas is"
graph_result = attribute(prompt, model, verbose=True)

# 3. Grab baseline activations (needed for supernode defaults + intervention math).
logits, activations = model.get_activations(prompt, sparse=True)

# 4. Build the intervention graph from a Neuronpedia annotation.
url = "https://www.neuronpedia.org/gemma-2-2b/graph?slug=gemma-fact-dallas-austin&..."
supernode_features = extract_supernode_features(url)

texas_node = Supernode(name="Texas", features=supernode_features["Texas"])
say_austin_node = Supernode(name="Say Austin", features=[Feature(23, 10, 12237)])

intervention_graph = InterventionGraph(
    ordered_nodes=[[texas_node], [say_austin_node]],
    prompt=prompt,
)
intervention_graph.initialize_node(texas_node, activations)
intervention_graph.initialize_node(say_austin_node, activations)

# 5. Intervene: ablate Texas at -2x.
Intervention = namedtuple("Intervention", ["supernode", "scaling_factor"])
intervention_tuples = []
for i, feat in enumerate(texas_node.features):
    default = texas_node.default_activations[i].item()
    intervention_tuples.append((*feat, -2.0 * default))

new_logits, _ = model.feature_intervention(prompt, intervention_tuples)

# 6. Compare top predictions.
top_baseline = logits[0, -1].topk(5)
top_ablated = new_logits[0, -1].topk(5)
print("Before:", [model.tokenizer.decode(t) for t in top_baseline.indices])
print("After: ", [model.tokenizer.decode(t) for t in top_ablated.indices])
```

Expected: baseline top prediction is `" Austin"`; after ablating Texas, `" Austin"` drops out and other capital-city tokens (or generic completions) take its place.

---

## Relationship to the manual pipeline in Section 3

Every circuit-tracer call maps onto something implemented by hand in [notes.md](notes.md):

| circuit-tracer | Section 3 manual equivalent |
|---|---|
| `ReplacementModel.from_pretrained(...)` | Load model + load transcoder stack + attach hooks |
| `attribute(prompt, model)` | `FreezeHooks` + `TranscoderReplacementHooks` + `build_graph_nodes` + `compute_adjacency_matrix` + `compute_influence` + prune |
| `model.get_activations(prompt, sparse=True)` | Forward pass + extract `hook_sae_acts_post` per layer |
| `model.feature_intervention(prompt, ivs)` | Forward pass with hooks that override `hook_sae_acts_post` at the specified indices |
| `model.feature_intervention_generate(...)` | Same as above but in a generation loop, with slice-position interventions |
| `create_attribution_dashboard(result, model)` | HTML rendering of the pruned adjacency matrix + example data |

If something in the library is confusing, the corresponding manual implementation in Section 3 is the ground-truth reference for what it's doing under the hood.

---

## See also

- The paper: [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (Ameisen et al., 2025). Establishes the linearized-model + gradient-attribution pipeline.
- The demo notebooks in the library repo: `circuit-tracer/demos/`. Additional worked examples beyond Dallas/Austin.
- [notes.md](notes.md) - the manual implementation of the same pipeline in Section 3 of the ARENA notebook.
- Neuronpedia's [graph explorer](https://www.neuronpedia.org/gemma-2-2b/graph) - browsable interactive versions of many pre-built Gemma graphs, with feature annotation and supernode pinning.
