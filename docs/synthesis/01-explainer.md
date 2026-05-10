# LARQL: A High-Level Explainer

> *"The model is the database."*

## What is this, really?

A trained transformer language model is conventionally treated as a black box: a
multi-gigabyte blob of floating-point weights that you load onto a GPU, feed
tokens through, and read tokens out of. Everything in between — every step of
"reasoning," every fact recall, every formatting choice — happens inside dense
matrix multiplications that are, in principle, opaque.

**LARQL inverts that picture.** It treats the trained transformer as the
compiled, encoded form of a graph of *facts and transformations*, and provides
tools to:

1. **Decompile** a model's weights into that graph form (a directory called a
   *vindex*),
2. **Query** the graph directly with a SQL-like language (LQL),
3. **Edit** the graph — insert new facts, delete or rewrite existing ones,
4. **Recompile** the modified graph back into standard model weights that any
   inference engine (HuggingFace Transformers, llama.cpp/GGUF, MLX) can run.

All without retraining and without a GPU.

The claim is strong, and the project backs it with measurable results: the
correctness of the decompilation has been verified at every layer boundary on
Gemma 3‑4B (zero divergence vs the dense forward pass), and a "training-free"
insert of the fact `(Atlantis, capital-of, Poseidon)` produces 94.6%
top-1 confidence on `"The capital of Atlantis is __"` while keeping
`(France → Paris)` at 60.5% — about a 20-point neighbour-degradation cost
for one new permanent fact, eight feature writes, no fine-tuning.

---

## The core insight

A modern transformer FFN block applies the operation:

```
y = W_down · ( SiLU(W_gate · x)  ⊙  (W_up · x) )
```

where `x` is the residual-stream vector at a given token, and `W_gate`,
`W_up`, `W_down` are the three trained weight matrices.

This single line hides the structure that LARQL exposes. Read the matrices
*by row/column* instead of *as monolithic blocks*, and the FFN turns into a
collection of independent **features**, each of which is just a triple
`(gate_row, up_row, down_column)`:

| Component | What it does | Graph reading |
|-----------|--------------|---------------|
| `gate_row[i]` | Detects whether a particular pattern is present in `x` | The **trigger** of feature `i` |
| `up_row[i]` | Scales how strongly feature `i` should contribute | The **gain** of feature `i` |
| `down_col[i]` | Direction added to the residual when feature `i` fires | The **write** of feature `i` |

Each FFN feature is therefore a **directed edge** in a graph: it reads a
pattern out of the residual stream and writes a vector back into it. The
*entire* FFN of a transformer is just an enormous unordered set of such
edges — for Gemma 3‑4B, **34 layers × 10,240 features = 348,160 edges**.

LARQL extracts those edges, indexes them, labels them by examining what
tokens they read and write (using the model's own embedding table), and
stores the result on disk as a *vindex*.

The vindex is the model. Not a summary of it, not a pruned version of it —
the *exact same numbers*, just laid out for queryability rather than dense
batched matrix multiplication.

---

## Two query primitives

Once the FFN has been re-conceptualised as a graph, two simple operations do
most of the work:

### 1. **Gate KNN** — "what fires at this layer for this input?"

Given a residual vector `q` at layer `ℓ`, compute the dot product against
every gate row in that layer and keep the top‑K. This is *literally* the
first half of the dense matmul (`W_gate · x`), but instead of multiplying
through every feature you stop after the K largest activations.

In Gemma 3‑4B, K ≈ 10–100 features per layer is enough to recover the dense
forward pass to within float-noise accuracy. The remaining ~10,000 features
contribute negligibly because their gate activation is small and the SiLU
non-linearity drives them to zero.

This is the *walk* operation in LARQL. It runs the FFN block as
`gate KNN → sparse down → add to residual` and produces results
indistinguishable from the dense forward pass. On a single CPU it actually
runs *slightly faster than dense* (517 ms vs 535 ms per token on Gemma 3‑4B)
because the walk reads from a feature-major mmap'd file with better
page-cache behaviour than row-major safetensors.

### 2. **Down KNN** — "what does this feature actually output?"

Given a feature's `down_column`, project it against the model's token
embedding table and keep the top‑K. The tokens that come back are *what
that feature is* in human-readable terms.

Apply this exhaustively across the whole FFN and you get labelled edges:

```
F5040  → French (also: french, FRENCH, France, Frenchman, француз, フランス)
F943   → euros  (also: €, Euros, EU, 欧盟, Spain, EUR)
F2230  → Dutch  (also: Netherlands, dutch, Amsterdam, 荷兰)
```

Cross-lingual knowledge surfaces automatically; no separate analysis needed.
The feature's down vector points into a *region* of embedding space that
spans many surface forms.

---

## What you can do with it

Once you have the vindex, queries become structural rather than statistical:

```sql
USE "gemma3-4b.vindex";

DESCRIBE "France";
-- France
--   Edges (L14-27):
--     capital     → Paris       1436.9   L27
--     language    → French        35.2   L24
--     continent   → Europe        14.4   L25
--     borders     → Spain         13.3   L18

WALK "The capital of France is" TOP 10;
-- per-layer feature scan; no forward pass needed

INFER "The capital of France is" TOP 3;
-- 1. Paris  (97.91%)   ← full forward pass with attention

TRACE "The capital of France is" FOR "Paris" DECOMPOSE LAYERS 22-27;
-- Layer  Rank  Prob   Attn    FFN   Who
--   L22    50  0.002  +22.2  +34.4  BOTH ↑
--   L23    10  0.024  -16.9  +55.9   FFN ↑
--   L24     1  0.714 +105.7  +24.4  BOTH ↑   ← phase transition
--   L26     1  0.999  +83.1  +18.7  BOTH ↑
```

The `DESCRIBE` is reading edges directly off the graph. The `WALK` runs the
gate-KNN scan layer by layer. The `INFER` runs a full forward pass — but
implemented as walk + standard attention, so all of it lives in the same
queryable substrate. The `TRACE` decomposes the residual stream into its
attention and FFN contributions, layer by layer, so you can see exactly
when and how an answer "crystallises."

This is mechanistic interpretability *as a database*, not as a
research notebook.

---

## Editing knowledge without retraining

The most striking capability is direct editing. Because the FFN is just a
collection of edges and the graph is loss-free with respect to the original
matrices, you can:

- **Insert** a new edge at chosen layers, with chosen confidence/strength.
- **Delete** an existing edge.
- **Update** the target of an existing edge.
- Save the changes as a *patch* (`.vlp` JSON file, ~10 KB per fact).

```sql
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital-of", "Poseidon");
-- Auto-patch started.

INFER "The capital of Atlantis is";
-- Poseidon (94.6%)
```

The technique used is *constellation insert*: rather than rewriting one
strong feature at one layer (which causes severe collateral damage —
`France → Paris` collapses), the change is spread across roughly eight
layers in the upper "knowledge band" of the model, each contributing a
small nudge (alpha ≈ 0.25) toward the target embedding. The cumulative
effect on the inserted fact accumulates because nothing else competes for
that exact prompt; the cumulative effect on neighbouring facts is absorbed
because they have strong existing signals (Paris) that the small nudges
cannot overpower.

The arithmetic is simple. The *reason it works* is the geometry of the
residual stream: the actual query vectors that the gate sees at layer 24+
are nearly orthogonal to the raw token embedding (`cos(embed("Atlantis"),
residual_at_L24) ≈ 0.01`). LARQL exploits this by capturing the *real*
residual at the relevant layer with a single forward pass, normalising it
to match the magnitude of existing gates, and using *that* as the trigger
of the new feature. Then the down vector is placed in the direction of
`embed("Poseidon")` so that the LM head's projection picks up the new
target token.

Patches stack, are reversible, and are tiny. A 1,000-fact domain patch is
about 10 MB compared to the full 8 GB model — roughly 1/800th the size,
and shareable as a JSON file.

---

## Recompiling back to standard weights

Once edits are applied, two compile targets are available:

1. **`COMPILE CURRENT INTO VINDEX`** — produces a fresh standalone vindex
   directory with the patches baked in. On APFS, base weight files are
   hardlinked from the source and only `down_weights.bin` is rewritten
   column-wise (instant on small models, milliseconds-to-seconds on large
   ones). Reading this vindex needs no overlay logic at load time.

2. **`COMPILE CURRENT INTO MODEL`** — produces standard `safetensors` (or
   GGUF) weight files that any external inference engine can load. The
   inserted edges live in the ordinary `down_proj` tensors of the
   resulting model, so HuggingFace Transformers, llama.cpp, or MLX
   pick them up with zero special handling.

The primitive at the bottom of all of this is a function called
`install_edge`
([`crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs:53`][edge]),
which does exactly one thing: given a free FFN slot `s` and a
`(trigger, write)` pair, it writes:

```
gate[s, :]  ←  trigger̂ × g_norm × gate_scale
up[s,   :]  ←  trigger̂ × u_norm
down[:, s]  ←  write × (d_norm / ‖write‖) × alpha_mul
```

Reference norms (`g_norm`, `u_norm`, `d_norm`) come from the original slot
so the magnitude regime is preserved; `gate_scale` (typically 30) makes
the gate fire decisively when the trigger appears; `alpha_mul` calibrates
the strength of the contribution. That single primitive, multiplied across
a constellation of layers, is enough to encode a new fact into raw
weights.

[edge]: ../../crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs

---

## Capabilities at a glance

| Capability | What you get | Notes |
|------------|--------------|-------|
| Read knowledge from any open-weight transformer | `DESCRIBE`, `WALK`, `SELECT` over edges | Works without a GPU; vindex is mmap'd |
| Run inference on the decompiled form | `INFER` runs walk-FFN + standard attention | Identical predictions to dense (verified at all 34 layer boundaries on Gemma 3‑4B) |
| Trace internal computation | `TRACE` decomposes residual stream into per-layer attention/FFN deltas | Reveals "phase transitions" where the answer crystallises |
| Insert new facts | `INSERT INTO EDGES (...)` | One forward pass + ~8 feature writes; ~94% confidence on new fact, ~20pt cost on near-neighbours (tunable) |
| Delete or rewrite facts | `DELETE`, `UPDATE` | Mark features as deleted; edits live in patch overlay |
| Share knowledge edits | `.vlp` JSON patches, ~10 KB/fact | Stack, reverse, distribute |
| Recompile to standard weights | `COMPILE INTO MODEL` | Output runs in any inference engine; no special loader |
| Mechanistic-interpretability primitives | hooks: capture, ablate, steer, patch, logit-lens | Both Rust and Python (PyO3) |
| Distribute across machines | HTTP / gRPC server, FFN sharding, MoE expert sharding | CPU-only expert servers viable |

A single 4 B-parameter model decompiles into roughly:

- **~3 GB** browse vindex (gate vectors + embeddings + down metadata)
- **~6 GB** inference vindex (above + attention + FFN weights)
- **~10 GB** all-level vindex (above + LM head + compile metadata)

with `f16` storage. Quantised vindexes (Q4_K) are smaller still.

---

## What it isn't

Three things to keep clearly in mind:

- **It's not a new model architecture.** The transformer is exactly the
  transformer. LARQL just changes the storage layout of the FFN weights
  and provides operations over that layout.

- **It's not lossless interpretability.** About 85% of FFN features are
  "dark space" — features whose down vectors do not align with any single
  token embedding. The current evidence is that these are *structural*
  computation (handling articles, formatting, scaling) rather than
  factual knowledge, so the 15% of features that *do* resolve cleanly
  appear to capture all the labelled facts. But the dark space is not
  yet fully understood.

- **It's not a training replacement** for everything. The constellation
  insert injects facts cleanly but with a measurable degradation cost on
  near-neighbours, and the technique has been validated for short
  factual triples — multi-token targets, long passages, and complex
  procedural skills are open research questions (the `experiments/`
  directory documents partial progress on each).

What it *is* is a faithful, queryable, editable representation of a
trained transformer's parametric knowledge — exact enough to round-trip
to inference output, structural enough to support graph operations.
That's a meaningful addition to the toolkit even before any of the more
speculative implications are pursued.
