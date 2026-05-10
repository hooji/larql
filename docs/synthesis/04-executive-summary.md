# LARQL: Executive Summary

## What it is

LARQL ("Lazarus") is a tooling stack that **decompiles a trained transformer
language model into a queryable graph database (a *vindex*) and provides a
SQL-like language (LQL) to browse, edit, and recompile that graph back into
standard model weights.** Decompilation is loss-free with respect to the
model's behaviour: the graph form runs inference and produces the same
predictions as the original dense matrices, validated at every layer
boundary on Gemma 3‑4B. Edits are persisted as small (~10 KB/fact) JSON
patches that overlay an immutable base; recompiling produces drop-in
`safetensors` (or GGUF) weights that any inference engine can load.

The technique is built on a clean mathematical insight: a transformer's
FFN block is mathematically identical to a sparse sum over independent
"features," each of which is a directed edge `(trigger, write)` between
points in residual-stream space. Ten thousand such edges per layer,
across thirty-odd layers, gives a graph of roughly 350 000 directed
edges per 4 B-parameter model — that graph *is* the model's parametric
knowledge.

## Why it matters

Three capabilities follow naturally from the decomposition, and each is
hard or impossible with the conventional "weights as opaque blob" view:

1. **Direct knowledge query.** `DESCRIBE "France"` returns labelled
   edges (`capital → Paris`, `language → French`, `borders → Spain`)
   read straight off the graph in milliseconds, on a CPU, no GPU
   required. Cross-lingual labels ("french / Frenchman / француз /
   フランス") surface automatically because the down vector points
   into a multi-language region of embedding space.

2. **Training-free fact insertion / deletion.** Adding
   `(Atlantis, capital-of, Poseidon)` takes one forward pass plus eight
   feature writes (~30 seconds total, no GPU). The new fact reaches
   ~94% top-1 confidence; near-neighbour facts (France→Paris) take a
   ~20-pt collateral hit, tunable down to ~14 pt by spreading the change
   across more layers at smaller per-layer strength. Deletion is
   immediate; patches stack and are reversible.

3. **Mechanistic interpretability as a first-class API.** `TRACE`
   decomposes a forward pass into per-layer attention and FFN
   contributions, revealing the "phase transition" layer where an
   answer crystallises. Hooks for capture, ablation, steering, logit-lens,
   and activation-patching are zero-cost when unused and exposed in
   both Rust and Python.

Throughout, the system stays inside the standard transformer family: no
new architecture, no retraining, no GPU requirement for the offline
operations. Inference uses BLAS-fused attention and walk-FFN, with a
Metal GPU path for production decode on Apple Silicon.

## Why it (probably) works

The FFN is a sum of independent feature contributions whose individual
magnitudes are heavy-tailed: a tiny fraction (≈10 of 10 240 features at
each layer in Gemma 3-4B) account for almost all of the per-token
output. The "walk" replaces the dense `H × I` matrix multiply with a
top-K gather-and-sum, recovers the original output to within
floating-point noise, and turns out to run fractionally *faster* than
the dense path on a single CPU thanks to better cache locality. About
85% of FFN features have down vectors that don't align with any single
token embedding — empirically these handle structural concerns
(formatting, scale, articles), so the 15% that *do* align cleanly
appears to capture all the model's labelled factual knowledge.

For inserts, the residual stream is additive across layers, so a small
nudge per layer in the direction of `embed(target) × √H` accumulates
into a strong logit shift on the trigger prompt while being absorbed by
strong existing signals on near-neighbour prompts. That's the
"constellation" technique, and its parameters are calibrated empirically
per model family.

## Implications for the LLM ecosystem

### For end users and small teams

A vindex is a queryable, editable representation of a model. Today this
means three concrete things:

- You can **browse the parametric knowledge** of any open-weight
  transformer (Gemma 2/3/4, Llama 2/3, Mistral, Mixtral, Qwen, Phi,
  DeepSeek, GPT-2, GPT-OSS) on a CPU laptop with no GPU.
- You can **distribute small knowledge patches** (`.vlp` files,
  ~10 KB/fact, ~10 MB/1,000 facts, vs ~8 GB for a 4 B-param model)
  instead of fine-tuned forks. This is approximately 1/800th the size
  of redistributing a full model.
- You can **compile patched vindexes back to safetensors** and run them
  in HuggingFace Transformers, llama.cpp, or MLX with no changes to
  those engines.

For specialised deployments (medical knowledge, company facts, fixing
a known hallucination), this is fine-tuning's value at orders-of-magnitude
less cost. For regulatory or compliance use cases, edits are auditable —
each patch is a JSON document.

### For large AI providers

The implications here are more nuanced; some are speculative.

- **A more direct interpretability surface for governance and safety
  work.** Reading "what does this feature output?" off the down
  vector — in human-readable token space, with multilingual coverage —
  scales to whole-model audits in a way that probing-based interpretability
  does not. The `TRACE` mechanism is a ready-made tool for understanding
  *where* in the depth dimension a behaviour comes from. This complements
  rather than replaces existing alignment research.
- **A surgical alternative to fine-tuning for narrow updates.**
  When the goal is to fix a specific factual error, suppress a specific
  hallucination, or inject a specific company-private fact, the
  constellation insert offers a documented, reversible, auditable
  alternative to RLHF or supervised fine-tuning. It is not a
  general-purpose replacement for fine-tuning — multi-token outputs,
  long passages, and procedural skills remain firmly in fine-tuning
  territory.
- **Distributed deployment topologies.** Because the vindex is an
  mmap'd directory, not a monolithic GPU residency, large models
  (26 B–1 T parameter range, MoE included) become deployable across
  CPU-only servers with the laptop running attention. The fly.io
  expert-server demo runs Gemma 4 26B-A4B on commodity CPU instances
  at a fraction of the GPU cost.
- **Speculative: weight generation rather than training.** The current
  `install_edge` primitive can write a fact directly into dense
  weights; if more general writing primitives are developed (multi-token
  outputs, procedural circuits, in-context routing), the GPU/memory
  cost of training small-to-medium models could in principle be
  substantially reduced. This is *not* yet a demonstrated capability
  for general training — it is a research direction the architecture
  enables.

The realistic short-term impact at provider scale is: better
interpretability tooling, surgical hot-fix capability, and cheaper CPU
inference for large MoE models. A more transformative impact (training
replacement, direct weight generation) is a research bet rather than a
shipping capability.

### For the open-source ecosystem

This is where the impact is most immediate.

- **A common decompiled format** (`vindex`) for any open-weight
  transformer makes interpretability research, weight comparison,
  and knowledge merging composable across model families.
- **Hot-fixable open models.** Hallucinations and stale facts in a
  released open-weight model can be addressed by community-maintained
  patch sets without re-training or re-publishing the model.
- **HuggingFace integration is already in place.** `larql publish`
  uploads vindexes (and slice variants) to HF and assembles
  collections; `larql pull` downloads with progress bars and resume
  semantics. The vindex format is BSD/Apache-2-clean and the
  reference implementation is Rust + PyO3.

### Honest limitations

1. **Validated mostly on Gemma family and similarly-shaped models.**
   The walk-correctness guarantee is per-architecture and per-family;
   MXFP4-quantized MoE (GPT-OSS) currently has degraded `DESCRIBE`/`WALK`
   results, with `INFER` as the supported path.
2. **Insert is short-fact only.** Multi-token targets, long passages,
   and procedural skills are open research questions. Subtoken targets
   ("Poseidon" → "Pose" first subtoken at 94.6%, "idon" must come from
   autoregressive continuation).
3. **Insert has measurable collateral cost.** A new fact at the
   recommended setting costs ~14–20 pt of confidence on near-neighbour
   facts. For high-value, low-volume edits this is acceptable; for
   thousand-fact corpora, capacity limits and re-balancing dynamics
   are still being characterised.
4. **About 85% of FFN features are not yet labelled.** They appear to
   be structural rather than factual, but a quantitative theory of
   the structural/factual split is still being developed.
5. **GPU dense path is still faster for raw decode.** On Apple
   Silicon, the Metal GPU dense path (~83 tok/s on Gemma 3-4B) beats
   the CPU walk (~2 tok/s). The walk's value is in queryability,
   editability, and CPU-only deployment, not in raw throughput.

## Bottom line

LARQL converts "model = opaque blob of weights" into "model = queryable
graph of facts plus standard transformer attention." That conversion
is loss-free, fast, runs on a laptop, and round-trips back to
standard weights. It already enables practical workflows
(knowledge browsing, surgical fact insertion, mechanistic interpretability)
that are awkward or impossible without it, and it opens a credible
research direction (direct weight generation as a training-cost
reducer) that should be pursued empirically. The technique is sound,
the implementation ships, and the limitations are documented and
tractable.

## One-line summary

> *A transformer is a graph of facts plus an attention router; LARQL
> exposes both halves so you can browse them, edit them, and write
> them back to disk.*
