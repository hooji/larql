# LARQL — A High-Level Explainer

> **The model IS the database.** A trained transformer's FFN is already a content-addressable lookup table. LARQL just rearranges the bytes so you can see it, query it, edit it, and recompile it — without GPUs and without retraining.

---

## 1. The core insight

A modern decoder-only transformer is built from a stack of two-thing layers:

- **Attention** — a router that decides *which past tokens* to look at.
- **Feed-forward network (FFN)** — a knowledge store that decides *what to add to the residual stream*.

For each FFN layer, the gated form (Gemma, Llama, Mistral, Qwen, Mixtral, …) is:

```
out = down( silu(x · W_gate) ⊙ (x · W_up) )
```

Now look at this as a graph problem instead of a matrix problem.

`W_gate` is a tall thin matrix of shape `[intermediate, hidden]`. Each of its `intermediate` rows is a single direction in the residual space — call it `g_i`. The dot product `x · g_i` measures *how strongly feature `i` recognizes the current state of the residual stream*.

`W_down` is `[hidden, intermediate]`. Its `i`-th column is the direction `d_i` that feature `i` writes back into the residual stream when it fires.

So feature `i` at layer `L` is exactly:

> "If the residual stream looks like `g_i`, push it in direction `d_i`."

That is a **labeled edge** in a graph:

```
trigger (gate)  ──[layer L, feature i, weight = silu(...)·(up·x)]──▶  output (down)
```

There are `num_layers × intermediate_size` such edges. For Gemma 3 4B that's 34 × 10,240 = **348,160 edges**. The full FFN is a graph with that many edges. **The model already is a graph; nobody had been treating it as one.**

LARQL takes that observation seriously. It pulls the gate rows out into a KNN index, stores the down columns alongside them, and lets you query and edit the graph directly.

---

## 2. The vindex

A **vindex** ("vector index") is a directory of memory-mapped files where the model's weights have been reorganized — not duplicated, *reorganized* — into the layout that supports graph queries.

```
gemma3-4b.vindex/
  gate_vectors.bin       # W_gate rows, layer-by-layer (the KNN index)
  embeddings.bin         # W_embed (token ↔ vector lookup)
  down_meta.bin          # Per-feature: the top-k vocabulary tokens its down vector points at
  attn_weights.bin       # Q, K, V, O per layer (only for INFER)
  up_weights.bin         # W_up per layer (only for COMPILE)
  down_weights.bin       # W_down per layer (only for COMPILE)
  norms.bin              # LayerNorm parameters
  index.json             # Config, layer bands, provenance, checksums
  tokenizer.json         # HuggingFace tokenizer
  relation_clusters.json # Discovered relation types
  feature_labels.json    # Probe-confirmed labels (e.g. "L26:F9298 = capital-of")
```

Three extraction levels gate which operations are usable:

| Level | Files | Size (4B, f16) | Operations enabled |
|-------|-------|---------------:|--------------------|
| **Browse** | gate + embed + down_meta | ~3 GB | DESCRIBE, WALK, SELECT |
| **Inference** | + attention + norms | ~6 GB | + INFER (full forward pass) |
| **All** | + up + down + lm_head | ~10 GB | + COMPILE back to safetensors / GGUF |

Crucially, `gate_vectors.bin` *is* `W_gate` — same float values, different file layout. COMPILE reads it back unchanged when reconstructing the model.

---

## 3. LQL — querying and editing the graph

LARQL ships **LQL** (Lazarus Query Language), a SQL-flavored language for the graph. The four headline verbs are:

```sql
-- decompile a HuggingFace model into a vindex
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex";

-- ask what the model knows about an entity
DESCRIBE "France";
--   capital     → Paris      1436.9  L27  (probe)
--   language    → French       35.2  L24  (probe)
--   continent   → Europe       14.4  L25  (probe)
--   borders     → Spain        13.3  L18  (probe)

-- write a new fact (no GPU, no fine-tuning)
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital-of", "Poseidon");

-- bake the edits into a real model file
COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;
```

`DESCRIBE` runs gate KNN at every knowledge-band layer for the entity's embedding and aggregates the top output tokens. The whole walk takes ~33 ms on a laptop CPU — no GPU required.

`INSERT` triggers a multi-step procedure (training-free knowledge insertion, see below) that synthesizes new (gate, up, down) vectors and stages them as a **patch** — a small JSON overlay that does *not* mutate the base vindex on disk.

`COMPILE INTO VINDEX` flattens active patches into a fresh vindex (instant on APFS via hardlinks; only `down_weights.bin` is rewritten in place). `COMPILE INTO MODEL` writes a standard HuggingFace safetensors directory whose tensors load cleanly into PyTorch / Transformers / GGUF runtimes — no special loader needed.

There are 20+ statements across five categories: lifecycle (EXTRACT, COMPILE, DIFF, USE), browse (DESCRIBE, WALK, SELECT), inference (INFER, EXPLAIN INFER), trace (TRACE … FOR Token), mutation (INSERT, DELETE, UPDATE, MERGE), patches (BEGIN/SAVE/APPLY/REMOVE PATCH), and introspection (SHOW LAYERS/FEATURES/MODELS/PATCHES, STATS).

---

## 4. Walking the FFN

The most striking demonstration that this view is *equivalent* to standard inference is the **walk-FFN**. Replace the dense `down(silu(x·W_gate) ⊙ (x·W_up))` with:

```
1. hits = gate_knn(layer, x, top_k)     # find features with largest |x · g_i|
2. for each (i, gate_score) in hits:
       up_score = x · u_i               # u_i = up vector for feature i
       a_i = silu(gate_score) · up_score
       out += a_i · d_i                 # d_i = down vector for feature i
```

This is *the same arithmetic* — just expressed as "look up the K active features, then sum their weighted down contributions" instead of "do three dense matmuls". The walk-FFN is **not an approximation** when `K ≥ intermediate_size`; it is bit-identical to dense up to floating point.

The empirical result is striking:

- A boundary sweep on Gemma 3 4B replaced dense FFN with walk-FFN for layers `B..34` (varying `B` from 0 to 34) and ran 5 canonical capital-of prompts. **At every boundary, top-1 token and probability were identical.** Zero divergence.
- With `top_k = 8092` (out of 10,240 features), walk-FFN runs in **517 ms** for a full Gemma 3 4B forward pass on M-series CPU vs **535 ms** for dense — *the walk is faster than the dense matmul*, because the down read from a feature-major mmap'd file has better cache behavior than the safetensors layout.

Two implications follow:

- **FFN is a content-addressable store, not a black-box matmul.** The K activated features at each layer are the model's "answer" to "what's salient in the residual right now?".
- **You don't need to materialize the dense matrix to run inference.** The mmap'd vindex *is* the FFN. The OS demand-pages whatever rows the gate KNN selects.

---

## 5. Editing the graph

Because each (gate, up, down) triple is the model's atomic unit of knowledge, you can write new triples directly. The procedure validated as `INSERT INTO EDGES` is:

1. **Capture residuals.** Run the model once on a canonical prompt like `"The capital of Atlantis is"`. Record the post-attention residual at each layer in the knowledge band (typically L20–L27 for Gemma 4B).
2. **Synthesize a slot per layer.** For each chosen layer:
   - `gate ← unit(residual_at_L) × g_norm × 30` — a vector that *fires when the captured residual appears*. The 30× makes it competitive with trained features.
   - `up ← unit(residual_at_L) × u_norm` — same direction as gate (so silu(gate·x) and up·x reinforce each other).
   - `down ← unit(embed("Poseidon")) × d_norm × α` — the answer direction, scaled to layer-typical magnitude.
3. **Refine via Gram-Schmidt.** Orthogonalize against decoy entities at the same layer (so "Atlantis" doesn't break "France"). This step produces the multi-fact constellation; without it later inserts contaminate earlier ones.
4. **Stage as a patch.** The triple lives in a `.vlp` JSON overlay; the base vindex bytes are unchanged.
5. **Optionally COMPILE.** Bake the override columns into a fresh `down_weights.bin`. The result is a standard model file; HuggingFace Transformers loads it and produces the new fact through the *standard* FFN path with no overlay code.

End-to-end measured on Gemma 3 4B:

```
Before INSERT:
  "The capital of Atlantis is" → "said" (17.8%)

After INSERT (8 layers × α=0.25, single forward pass + 8 feature writes):
  "The capital of Atlantis is" → "Pose" (94.6%)         ← new fact

  "The capital of France is"   → "Paris" (60.5%)        ← preserved (was 80.5%)
```

A single fact patch is ~30 KB. A 1,000-fact domain patch is ~30 MB. Compared to a LoRA adapter (50–200 MB) or a full model (8 GB), that's two to three orders of magnitude smaller, and unlike LoRA you can read it as JSON.

---

## 6. The capabilities you get

Concretely, treating the model as a graph database unlocks operations that don't exist in the standard inference framework:

### Browse
- **`DESCRIBE "France"`** — list every (relation, target, layer, score) edge associated with an entity. Pure dot products. ~33 ms on a laptop CPU. No forward pass.
- **`WALK "The capital of France is" TOP 10`** — show which FFN features fire for a prompt's last-token residual at each layer. The factual answer surfaces as features at L26-27.
- **`SHOW FEATURES AT LAYER 26`** — enumerate features by label, with cluster types and probe confirmations.

### Inference (with model weights)
- **`INFER "The capital of France is" TOP 5`** — full forward pass, but the FFN runs through the walk path (mmap'd, sparse).
- **`TRACE "The capital of France is" FOR "Paris"`** — per-layer answer trajectory: rank, probability, attention contribution, FFN contribution, and which mechanism is doing the work at each layer.

### Mutation
- **`INSERT INTO EDGES (entity, relation, target) VALUES (…)`** — training-free knowledge insertion via the constellation method. Auto-creates a patch overlay.
- **`DELETE FROM EDGES WHERE entity = … AND relation = …`** — remove edges, validated by re-running INFER.
- **`UPDATE EDGES SET target = "London" WHERE entity = "John" AND relation = "lives-in"`** — change an edge in place.
- **`MERGE`** — combine knowledge from two vindexes.

### Compile
- **`COMPILE CURRENT INTO VINDEX "out.vindex"`** — bake patches into a standalone vindex. Hardlinks unchanged files (instant on APFS); only rewrites `down_weights.bin` columns at edited slots.
- **`COMPILE CURRENT INTO MODEL "out/" FORMAT safetensors`** — emit a standard HuggingFace model. Loads in PyTorch / Transformers / vLLM / Ollama / llama.cpp without modification.

### Distribute
- **`larql publish`** — upload to HuggingFace Hub as full vindex + sliced siblings (client / server / browse / attn / embed) + collections.
- **`larql serve --port 8080`** — HTTP + gRPC server. `serve --ffn-only --layers 0-19` shards by layer; `--experts 0-63` shards MoE expert banks across multiple CPU-only servers. The "laptop runs attention, beefy server holds FFN" topology scales from 4B models on a laptop to 1T-parameter MoE across a CPU grid.

### Mechanistic interpretability
- **Forward hooks** — `RecordHook`, `ZeroAblateHook`, `SteerHook`, activation patching, full logit lens, embedding-neighbor lookup, KV-cache surgery. Zero overhead when no hook is registered.
- **Circuit type classification** — `cos(gate_i, down_i)` partitions every feature into projector (factual bridge), identity (self-reinforce), inverter (suppression), transform (morphological/syntactic), suppressor. Reveals the model's three computational phases (computation L7–L18, knowledge L19–L29, format gate L30–L33 on Gemma 3 4B).
- **Trace / residual-stream decomposition** — every layer's attention delta and FFN delta, additively reconstructable, mmap-storable. Enables tiered context (3,000× compression vs KV cache).

---

## 7. Why this is a big deal

Three things change qualitatively when you treat the model as a graph database:

1. **Inspectability.** A 4B-parameter model becomes a 348,160-edge graph you can browse with DESCRIBE in milliseconds, on a CPU, with no GPU and no Python. The unit of knowledge is the (gate, down) pair — a thing humans can label and reason about.

2. **Editability without retraining.** Inserting a new fact takes one forward pass plus eight feature writes (~30 ms total) and produces 94.6% confidence on the new fact while preserving 60.5% confidence on the old neighbour. No GPU, no fine-tuning, no hours of compute. The 30 KB JSON patch survives `COMPILE INTO MODEL` and loads in any standard runtime.

3. **Distributability.** The vindex format separates weights by *function*, not by file size. The "browse" slice is 3 GB; the "client" slice (attention, embeddings, norms) is 7 GB at Q4_K; the "FFN server" slice can be hosted on CPU-only commodity machines and accessed over HTTP. A 1T-parameter Kimi-K2.6 / DeepSeek V4-class model becomes a multi-shard CPU deployment instead of an H100 cluster.

The combination is unusual: a research-grade interpretability surface (vindex + LQL + TRACE), a production-grade inference engine (walk-FFN faster than dense, 19 tok/s on 26B-A4B Metal, CPU-only MoE serving), and a write-back path that lets edits flow round-trip into standard model formats. The traditional ML stack has these as separate worlds; LARQL collapses them into one.

---

## 8. Where to dig deeper

- **Math foundations:** [`02-mathematical-foundations.md`](02-mathematical-foundations.md) — the matmul-to-graph identity, why walk = dense, the constellation method derivation.
- **Software architecture:** [`03-software-architecture.md`](03-software-architecture.md) — crate layout, pipelines, key subroutines, future directions.
- **Executive summary:** [`04-executive-summary.md`](04-executive-summary.md) — 1-2 page brief on impact for consumers, large labs, open source.
- **Q&A and toy implementation:** [`05-qa-and-toy-implementation.md`](05-qa-and-toy-implementation.md) — answers to the framing questions plus a Java sketch for nanochat / GPT-2.
- Existing internal docs that informed this: [`format.md`](format.md), [`ffn-graph-layer.md`](ffn-graph-layer.md), [`walk-boundary-sweep.md`](walk-boundary-sweep.md), [`training-free-insert.md`](training-free-insert.md), [`circuit-types.md`](circuit-types.md), [`residual-trace.md`](residual-trace.md).
