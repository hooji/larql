# LARQL — Software Architecture

> A walkthrough of the current LARQL codebase: how the crates compose, what each module owns, the core sub-routines, and where the most useful improvements are likely to land. Targeted at someone evaluating or extending the system.

---

## 1. Cargo workspace layout

LARQL is a Rust Cargo workspace. Two crate families:

```
LARQL-specific (depend on vindex / LQL / inference):

larql-models       Architecture detection, weight loading, quant/dequant kernels.
    ↓
larql-compute      CPU + Metal matmul backends, FFN kernels, MoE prefill.
    ↓
larql-vindex       Vindex lifecycle: extract, load, query, mutate, patch, save, Vindexfile.
    ↓
larql-core         Graph algorithms (BFS, PageRank, shortest-path, merge, diff).
larql-inference    Forward pass, BLAS-fused attention, Metal GPU, WalkFfn, hooks, trace.
    ↓
larql-lql          Lexer / parser / executor / REPL + USE REMOTE client.
    ↓
larql-server       HTTP + gRPC: serve a vindex, FFN-only mode, expert sharding.
larql-router       Static layer-range router.
larql-experts      MoE expert primitives.
larql-cli          The `larql` binary; commands/ holds one file per subcommand.
larql-python       PyO3 bindings — module name `larql._native`. Maturin-built.
kv-cache-benchmark Standalone benchmark crate.

Portable (no LARQL deps):

model-compute      Bounded native kernels (arithmetic / datetime) + optional
                   wasmtime-hosted WASM modules. Used at compile-time to resolve
                   computed answers (e.g. sum(1..101)). Knows nothing about
                   vindex or LQL. Will graduate to a sibling repo eventually.
```

The dependency flow is one-way; `model-compute` never imports `larql-*`.

The CLI is a thin dispatcher: each `larql <cmd>` lives in `crates/larql-cli/src/commands/{primary,extraction,query,dev}/<cmd>.rs` and is wired into the `Commands` enum in `crates/larql-cli/src/main.rs`. `larql serve` execs into `larql-server`. `larql repl` and `larql lql` delegate to `larql_lql::run_repl` / `run_statement`.

---

## 2. larql-models — the architecture / weights layer

Owns all knowledge of how a transformer is shaped on disk:

- **Architecture detection** — `detect_architecture_validated()` sniffs the safetensors directory + `config.json`, returns a `Box<dyn ModelArchitecture>` for one of: Gemma 2/3/4, Llama 2/3, Mistral, Mixtral, Qwen 2/2.5, Phi 2/3, DeepSeek V2/V3, GPT-2, GPT-OSS.
- **Tensor naming** — `key_prefixes_to_strip()`, `ffn_gate_key(layer)`, `attn_q_key(layer)` etc. Rather than every consumer hard-coding `model.layers.{L}.mlp.gate_proj.weight`, the architecture trait owns the mapping.
- **Quantization codecs** — `quant::fp4`, `quant::fp8`, `quant::q4k`, `quant::q6k`, `quant::half (f16)`, MXFP4 dequant. Block layouts, manifests, `row_dot` / `bytes_per_row` utility functions.
- **Loading helpers** — `ModelWeights::load_streaming(...)` mmaps safetensors and gives lazy access. `tensors`, `vectors` HashMaps for dense and 1-D tensors. Per-layer norm parameters.
- **Architecture-specific quirks** — Gemma 4 E2B per-layer embeddings, Gemma 2/3 final logit softcap, Gemma sliding-window attention, GQA ratios, partial RoPE, cross-layer KV sharing.

This is the "boring" layer that absorbs vendor heterogeneity. Adding a new model family lives almost entirely here — once `larql-models` knows the names and shapes, downstream code just sees `ModelWeights`.

---

## 3. larql-compute — the kernel layer

Owns the bandwidth-critical primitives:

- **MatMul backend trait** — `MatMulBackend::matmul`, `matmul_transb`, `matmul_batch`. Implementations: CPU (Accelerate / OpenBLAS via ndarray's `.dot()`), Metal GPU (32×32 tiled compute shaders, weight-buffer cache).
- **Auto-calibration** — at startup, benchmark CPU vs Metal at representative sizes (attention projections, FFN layers). Compute the FLOP threshold above which Metal wins. Used to route small ops (QK^T at 18K FLOPs) to CPU and big ops (FFN gate at 315M FLOPs) to Metal.
- **Q4_K / Q6_K kernels** — block-quantized matmul, `q4k_matvec` for sparse FFN, `q4k_ffn_gate_up_8sg` and `geglu_gelu_tanh` Metal compute shaders.
- **Fused kernels** — `silu_gate_up`, `gelu_tanh_gate_up`, RMS norm + scale, RoPE, fused attention block.
- **MoE dispatch** — `moe_dispatch.rs` for top-K expert routing, parallel collection, gRPC fan-out for remote experts.
- **NEON / AVX2 paths** — for ARM and x86 CPUs without GPU.

The auto-calibrating hybrid dispatch is the key UX win: there are no magic constants. Larql figures out the right routing on first run.

---

## 4. larql-vindex — the storage and graph layer

The largest crate. Organised into:

```
crates/larql-vindex/src/
  config/        VindexConfig, layer info, dtype, extract level
  format/        Filenames, weight manifest, quant manifest, bin format readers/writers
  index/         The VectorIndex itself
    core.rs        Top-level struct (composes substores)
    types.rs       FeatureMeta, GateIndex trait, WalkHit, callbacks
    storage/       gate_store, ffn_store, projection_store, metadata_store
    compute/       gate_knn (dispatch + scores_batch + hnsw_lifecycle), q4k_dispatch, router
    mutate/        Insert/delete/update slot mutations
  patch/         VindexPatch JSON, PatchedVindex overlay, LSM-style storage engine
  engine/        StorageEngine + epoch + MEMIT cycle (for COMPILE INTO MODEL closed-form edits)
  extract/       BuildContext pipeline: streaming, build_helpers, resume, stage_labels
  clustering/    Relation cluster discovery
  quant/         registry.rs (lookup quant_format → codec functions)
  vindexfile/    Dockerfile-like declarative vindex builds
  describe.rs    DESCRIBE algorithm (multi-layer KNN + label resolution)
```

### 4.1 The `VectorIndex` and substores

A `VectorIndex` composes four substores:

- **`gate: GateStore`** — the gate matrix mmap, decode caches, HNSW lifecycle, per-(layer, expert) caches.
- **`ffn: FfnStore`** — FFN mmap handles + Q4_K dequant cache + FP4 storage.
- **`projections: ProjectionStore`** — `lm_head` and attention weight mmaps.
- **`metadata: MetadataStore`** — `down_meta` + per-(layer, feature) overrides.

The substore split exists so adding a new field is a single edit in one substore, not a global cascade. Storage modes (heap, mmap, Q4_K, FP4) are distinguished by which fields inside `gate` / `ffn` are populated rather than a top-level discriminator.

### 4.2 The `GateIndex` trait

Both `VectorIndex` and `PatchedVindex` implement `GateIndex`:

```rust
trait GateIndex {
    fn gate_knn(&self, layer, residual, top_k) -> Vec<(usize, f32)>;
    fn feature_meta(&self, layer, feature) -> Option<FeatureMeta>;
    fn num_features(&self, layer) -> usize;
    fn down_override(&self, layer, feature) -> Option<&[f32]>;
    fn up_override(&self, layer, feature) -> Option<&[f32]>;
    fn has_overrides_at(&self, layer) -> bool;
    fn gate_knn_batch(&self, layer, x, top_k) -> Vec<usize>;
    // ... per-format ffn_row_dot / ffn_row_scaled_add etc.
}
```

Consumers like `WalkFfn` work transparently with patched or unpatched indexes — INSERT/DELETE/UPDATE immediately affect KNN results and inference output. Phantom-typed trait objects are used so the same forward-pass code runs over native f32, Q4K, FP4, and patched vindexes.

### 4.3 Gate KNN dispatch

`crates/larql-vindex/src/index/compute/gate_knn/`:

- **`dispatch.rs`** — top-level `gate_knn`, `gate_walk`, `gate_knn_expert`, `gate_knn_batch`, `gate_knn_q4`. Picks between BLAS, HNSW, GPU full-batch, and Q4 backend matvec.
- **`scores_batch.rs`** — full-batch BLAS / GPU matmul paths (`gate_scores_batch`, `gate_scores_2d_*`).
- **`hnsw_lifecycle.rs`** — HNSW enable/disable, lazy + eager build, per-layer + per-(layer, expert) caches.
- **`mod.rs`** — `top_k_by_abs` shared utility (binary heap of capacity K, O(K) extra memory regardless of N).

The path priority for a single KNN call:

1. **HNSW** (if enabled and a graph exists for this layer).
2. **f32 mmap zero-copy gemv** — slice directly out of the mmap, BLAS gemv. No allocation.
3. **f16 mmap decode-then-gemv** — read f16 bytes, decode to a stack/heap f32 vector, BLAS gemv.
4. **Q4K direct matvec** — backend kernel reads Q4K bytes, does the matvec without dequant.
5. **Heap fallback** — when the gate has been mutated by INSERT and not yet rebuilt to mmap.

### 4.4 Streaming extract

`extract/streaming.rs` mmaps safetensors shards and processes one layer at a time:

```
peak_memory = embeddings + 1 layer's gate + 1 layer's down + scratch
```

For a 120B MoE model, that is ~2 GB instead of ~120 GB. The extraction pipeline has 6 stages (resumable):

1. **Loading** — mmap safetensors, detect architecture.
2. **Embeddings** — write `embeddings.bin`, compute `down_meta.bin` (top-K vocab projection per feature).
3. **Gate** — copy `W_gate` rows per layer into `gate_vectors.bin` (or `interleaved_q4k.bin` for quantized). For MoE, experts are concatenated within each layer.
4. **Inference weights** (only at level Inference / All) — attention Q/K/V/O, norms.
5. **Compile weights** (only at level All) — up, down, lm_head.
6. **Manifests + checksums** — `index.json`, `weight_manifest.json`, SHA256s.

Resume is by file existence + `index.json` partial state. `--resume` skips stages whose outputs are present and intact.

### 4.5 Patches

A patch is a JSON file (`.vlp`) recording INSERT / DELETE / UPDATE operations. The on-disk format:

```json
{
  "version": 1,
  "base_model": "google/gemma-3-4b-it",
  "base_checksum": "...",
  "operations": [
    {
      "op": "insert", "layer": 26, "feature": 8821,
      "relation": "side_effect", "entity": "aspirin", "target": "bleeding",
      "confidence": 0.85,
      "gate_vector_b64": "<base64 f32 × hidden_size>",
      "up_vector_b64":   "<base64 f32 × hidden_size>",
      "down_vector_b64": "<base64 f32 × hidden_size>",
      "down_meta": {"t": "bleeding", "i": 12847, "c": 4.2}
    },
    {"op": "delete", "layer": 24, "feature": 1337, "reason": "hallucinated"}
  ]
}
```

A `PatchedVindex` overlays a base `VectorIndex`:

```rust
struct PatchedVindex {
    base: VectorIndex,
    patches: Vec<VindexPatch>,                       // applied in order
    overrides: HashMap<(usize, usize), PatchOp>,     // (layer, feature) → op
}
```

Calls to `gate_knn` etc. check overrides first, then fall through to the base. `bake_down()` flattens overrides into a fresh `VectorIndex` for save.

### 4.6 The storage engine (LSM-tier)

For multi-fact COMPACTion the `engine/` module implements an LSM-like tier discipline:

- **L0** — KNN journal: each `INSERT MODE KNN` entry is `(key_vector, target_id, confidence)` in `knn_store.bin`. Cheap to install, lookup at inference.
- **L1** — Compose entries: each `INSERT MODE COMPOSE` produces a `(gate, up, down)` triple and lives in the patch overlay.
- **L2** — Materialized weights: after `COMPACT MAJOR`, L1 compose entries are absorbed into `down_weights.bin` via column rewrite (or full MEMIT solve for stronger semantics).

Promotion verbs:

- `COMPACT MINOR` — promote L0 KNN to L1 compose (synthesize gate/up/down from the cached key vector + target embedding).
- `COMPACT MAJOR` — promote L1 to L2 (materialize into weight bytes). `WITH LAMBDA` knob for MEMIT regularization.

The point: the user shouldn't have to choose between "fast install" and "permanent weight edit" — they install fast, defer the commit, and amortize the math.

---

## 5. larql-inference — the forward-pass and hooks layer

### 5.1 The forward-pass shapes

```
crates/larql-inference/src/
  backend/                MatMulBackend selection, calibration
  attention/              GQA attention, BLAS-fused, online softmax
  ffn/
    weight.rs               WeightFfn — dense FFN from safetensors
    sparse_compute.rs       Shared sparse FFN compute (used by walk fallback)
    activation.rs           silu_gate_up, gelu_tanh_gate_up
    ...
  vindex/
    walk_ffn/               WalkFfn — sparse FFN via vindex gate KNN
      mod.rs                  Top-level WalkFfn struct + dispatch
      sparse.rs               Per-feature loop (the canonical baseline)
      exact.rs                K=m fast path (full-batch gemv)
      interleaved.rs / interleaved_q4k.rs / interleaved_q4.rs   Q-quant variants
      full_mmap.rs            All weights from feature-major mmap
      helpers.rs              Path selection helpers
    q4k_forward/            Q4K-aware forward pass (mech-interp friendly)
    l1_cache.rs             FFN activation cache (paraphrase collapse)
  forward/                Forward pass orchestration: trace_forward, predict, generate
  trace/
    capture.rs              Decomposed forward pass recording attn/FFN deltas
    store.rs                Mmap'd append-only binary store
    boundary.rs             Boundary residual store (10 KB / window)
    context.rs              Tiered context store
    types.rs                TraceNode, AnswerWaypoint, LayerSummary
  walker/                 Weight-level graph walkers (no forward pass — for BFS / probes)
  capture.rs              Mech-interp hooks (RecordHook, SteerHook, etc)
```

### 5.2 WalkFfn paths

For one FFN call at one layer there are five paths, picked by feature availability:

| Path | When | Down source | Speed (Gemma 4B) |
|------|------|-------------|------------------|
| **`exact`** (full-K gemv) | `K ≥ intermediate_size` and gated FFN | Full-batch BLAS / Q4K matmul | ~2 ms / layer |
| **`sparse`** | Default; `K < intermediate_size` | Per-feature down row from native mmap, Q4K, or FP4 | ~5–6 ms / layer |
| **`parallel_q4k_down`** | K ≥ 512 and Q4K storage with no native | Decoded down cache + rayon partial sums | ~5 ms / layer |
| **`full_mmap`** | All projections in feature-major mmap | All from mmap; slower due to TLB pressure | ~6.8 ms / layer |
| **`dense_fallback`** | Vindex doesn't have FFN data for this layer | Standard `down(silu(gate ⊙ up))` from safetensors | 6.7 ms / layer |

The dispatch is unified through the `GateIndex::ffn_row_dot` / `ffn_row_scaled_add` trait methods which route per-row through FP4 → native f32 → Q4K. **The same WalkFfn code runs over any storage format**; adding a new format is one impl block and one entry in the `quant::registry`.

### 5.3 BLAS-fused attention

Standard transformer attention computes `Q · Kᵀ → softmax → · V`, allocating a `[seq, seq]` matrix. The fused implementation:

```
for each query position qi, each head:
    scores[0..=qi] = K[0..=qi] · Q[qi]    # one BLAS gemv
    apply scale + softcap
    online softmax (max → exp → normalize, f64 accumulation)
    output = V[0..=qi]ᵀ · softmax_scores  # one BLAS gemv
```

Two `cblas_sgemv` calls per `(qi, head)`. No `[seq, seq]` allocation — the temp buffer is `O(seq)` per position. At seq=2048 with 10 heads that's 4 MB instead of 160 MB. The fused path runs at **~1.6× the materialized path** at the actual Gemma head dim of 256.

### 5.4 Hooks (mechanistic interp)

A simple trait fires at five intervention points per layer:

```rust
trait LayerHook {
    fn on_pre_layer(&mut self, layer, &h);                 // read-only
    fn on_attention_weights(&mut self, layer, &w);         // read-only (if capture)
    fn on_post_attention(&mut self, layer, &mut h);        // mutate
    fn on_ffn_activation(&mut self, layer, &gate);         // read-only (if capture)
    fn on_post_layer(&mut self, layer, &mut h);            // mutate
}
```

Built-in hooks: `NoopHook` (zero-cost when nothing registered), `RecordHook`, `ZeroAblateHook`, `SteerHook`, `CompositeHook`. Activation patching, full logit lens, embedding-neighbor lookup, KV-cache surgery are layered on top.

The single-forward path (`trace_forward_full_hooked`) is hookable; the multi-token Metal-fused decode path (`predict`) deliberately is *not* — kernels are fused and threading hooks through them would split the fast path even when no hook is registered. Hook-active generation runs through the CPU KV-cache path.

---

## 6. larql-lql — the parser and executor

### 6.1 Parser / AST / Executor symmetry

`crates/larql-lql/src/parser/` and `executor/` have matching `lifecycle.rs`, `query.rs`, `mutation.rs`, `introspection.rs`, `trace.rs`, `patch.rs`. When adding a statement: touch the AST in `ast.rs`, then both sides.

The lexer is hand-rolled (no `nom` / `pest`). The parser is recursive-descent. No grammar generator; the language is small enough that the LL(1) hand-roll is faster to debug than a generated parser would be.

### 6.2 Session and backend

The executor holds a `Session` with:

- **`backend: Backend`** — `Vindex(VectorIndex)`, `PatchedVindex(PatchedVindex)`, `Model(ModelWeights)` (for `USE MODEL`), `Remote(RemoteClient)` (for `USE REMOTE url`).
- **`raw_install_residuals: HashMap<(layer, feature), Array1<f32>>`** — uncontaminated captured residuals from each INSERT, used for batch-refine (Section 6.4 of the math foundations).
- **`decoy_residual_cache: HashMap<layer, Vec<Array1<f32>>>`** — canonical bleed-target prompts for each layer, used by Gram-Schmidt refinement.
- **`active_patch: Option<PatchSession>`** — the auto-created patch overlay for INSERT/DELETE/UPDATE.

### 6.3 INSERT execution

`crates/larql-lql/src/executor/mutation/insert/` is split into:

- **`plan.rs`** — `InstallPlan`: chosen layers, target embedding, target token id.
- **`capture.rs`** — Phase 1: run `infer_trace` on the synthesized prompt (`"The {relation} of {entity} is"`), capture per-layer residuals.
- **`compose.rs`** — Phase 2: walk the plan's layers, synthesize gate/up/down, refine via Gram-Schmidt against decoys + peer raw residuals.
- **`balance.rs`** — Phase 3: adjust down magnitudes to hit a canonical-prompt probability band (so the install doesn't over- or under-shoot).
- **`knn.rs`** — Alternative path for `MODE KNN`: just stores the residual + target as a retrieval entry.

Compose mode produces a complete (gate, up, down) triple per slot and writes all three into the `.vlp` patch — so a save → load → COMPILE round-trip is lossless.

### 6.4 COMPILE execution

`crates/larql-cli/src/commands/extraction/compile_cmd/` is the actual COMPILE machinery (the LQL `COMPILE INTO …` statements delegate here). Five files:

- **`mod.rs`** — top-level entry.
- **`detect.rs`** — sniff source format (vindex vs safetensors).
- **`single.rs`** — single-fact `larql compile --prompt … --answer …` from CLI.
- **`patch.rs`** — multi-fact COMPILE driven by an active patch overlay.
- **`save.rs`** — write outputs (vindex hardlink + column-rewrite, or safetensors emission).
- **`edge.rs`** — the `install_edge` primitive: write one `(gate, up, down)` triple at a slot with norm preservation and gate-scale conventions. **This is the lowest-level building block of the entire write-back path** and is the file that would be the first one extracted when a second consumer (TinyModel, a Java port) needs it.

---

## 7. larql-server — distribution

A minimal HTTP + gRPC server, but with several notable engineering details:

- **Layer / expert sharding** — `--layers START-END`, `--experts START-END` carve a vindex into a deployment slice. A laptop can hold attention + embed + router (~7 GB at Q4K for 31B); CPU-only commodity machines hold expert banks (~24 GB at Q4K for 26B-A4B) and serve `/v1/expert/batch`.
- **Slice presets** — `client`, `server`, `browse`, `attn`, `embed`, `router`, `expert-server`. `larql slice --preset client -o gemma3-4b.client.vindex` copies just the slices needed for a given role.
- **Bounding RSS** — `--ffn-only` skips the eager gate warmup (55 GB → 5.6 GB on 31B Q4K). `--max-gate-cache-layers 4` LRU caps decoded f16 gate heap. `--release-mmap-after-request` `madvise(DONTNEED)`s after each request (strict on Linux, advisory on Darwin).
- **Q4K wire format** — `/v1/walk-ffn-q8k` endpoint for streaming forward passes. Client sends Q8K-quantized residuals; server's NEON / Metal kernels compute FFN; small response.
- **Self-assembling gRPC grid** — multiple servers introduce themselves to a `larql-router` via gossip; the router builds a route table and the client just sends `--ffn http://router:9090`.
- **`fly.io`-tested** — reference deployment uses `deploy/fly/Dockerfile`. First boot pulls the vindex from HuggingFace to a persistent volume.

The split lets the model live where its weights make sense: laptop attention (latency-sensitive, small), CPU servers for FFN (memory-bound, cheap), beefy GPU server only for batch decode where it pays.

---

## 8. larql-python — bindings

Built with PyO3 + maturin under uv. Module: `larql._native`. The Python `larql` package (`crates/larql-python/python/larql/`) is the friendly façade.

Key exports:

- **`larql.load(path) → Vindex`** — knowledge queries, mutations.
- **`larql.WalkModel(path, top_k) → WalkModel`** — Rust inference with mmap'd weights. RSS for a 120B model: ~1 GB instead of 220 GB.
- **`larql.session(path) → Session`** — LQL session (`session.query("DESCRIBE 'France'")`).
- **`larql.mlx.load(path)`** — MLX model from vindex (Apple Silicon GPU, all weights in MLX memory).
- **`larql.walk_ffn.load(path, top_k)`** — MLX attention + Rust walk FFN (FFN mmap'd, only touched pages loaded).

Mech-interp surface (numpy in / numpy out):

```python
wm = larql.WalkModel("gemma3-4b.vindex")
residuals = wm.capture_residuals(prompt, layers=[12, 18, 24])
top5 = wm.logit_lens(residuals[24], k=5)
text, ids = wm.generate_with_hooks(prompt, max_new_tokens=10,
                                   steers=[(20, direction, 1.5)])
```

---

## 9. Cross-cutting subroutines

A short list of the most reused routines, with where they live:

| Routine | Crate / file | Purpose |
|---------|--------------|---------|
| `gate_knn(layer, x, K)` | `larql-vindex/src/index/compute/gate_knn/dispatch.rs` | The graph "address lookup". The atomic browse / walk operation. |
| `walk_ffn_sparse(layer, x)` | `larql-inference/src/vindex/walk_ffn/sparse.rs` | The graph traversal at one layer. Returns `(out, full_activation)`. |
| `install_edge(tensors, gate_key, up_key, down_key, slot, trigger, write, gate_scale, alpha_mul)` | `larql-cli/src/commands/extraction/compile_cmd/edge.rs` | Write one `(gate, up, down)` triple with norm preservation. |
| `install_compiled_slot(layer, residual, target_embed, alpha_mul, …)` | `larql-lql/src/executor/mutation/insert/compose.rs` | Phase 2 of INSERT — synthesizes a slot, calls install_edge under the hood. |
| `refine_layer_from_raw(patched, layer, raw_residuals, decoys, …)` | `larql-vindex/src/index/refine.rs` | Modified Gram-Schmidt over peer raws + decoys; rebuilds gates after each insert. |
| `bake_down()` | `larql-vindex/src/patch/core.rs` | Flatten patches into a fresh `VectorIndex`. |
| `compile_into_vindex(...)` | `larql-cli/src/commands/extraction/compile_cmd/save.rs` | Hardlink + down-column rewrite. |
| `trace_forward_full_hooked(weights, tokens, capture_layers, …, &ffn, &mut hook)` | `larql-inference/src/forward.rs` | Forward pass with five hook points per layer. The mech-interp foundation. |
| `gqa_attention_with_weights(...)` | `larql-inference/src/attention.rs` | Single shared attention function used by every forward path (dense, walk, trace, hooked). |
| `top_k_by_abs(scores, K)` | `larql-vindex/src/index/compute/gate_knn/mod.rs` | O(K) extra-memory top-K via min-heap. |

---

## 10. Testing and benchmarks

LARQL ships **2,000+ tests** across the workspace, with `make ci` running fmt + clippy with `-D warnings` + the test suite. Highlights:

- **Per-crate test counts** — larql-lql 272, larql-vindex 600+, larql-inference 109 (+6 with `--features metal`).
- **Boundary sweep** (`larql-inference/examples/walk_boundary_sweep.rs`) — exercises walk-FFN at every layer-cut from L0 to L34. Top-1 must match dense everywhere.
- **Compile demo** (`larql-lql/examples/compile_demo.rs`) — `INSERT Atlantis → Poseidon`, `COMPILE INTO VINDEX`, fresh `USE`, verify both Atlantis and France via INFER.
- **Refine demo** (`larql-lql/examples/refine_demo.rs`) — 10-fact INSERT + COMPILE; expected 10/10 retrieval on canonical prompts.
- **Memit decomposition demo** (`larql-vindex/examples/demo_memit_solve.rs`) — closed-form weight editing round-trip.

Criterion benches in each crate (`cargo bench -p larql-lql --bench parser`, `… executor`, `… compile`; `larql-vindex --bench vindex_ops`, `vindex_scaling`, `memit_solve`, `extract_throughput`, `q4k_vs_f32`; `larql-compute --bench matmul`).

The boundary sweep is the single most important correctness anchor in the codebase: it proves the walk-FFN substitution is faithful at the level of model output, not just intermediate residuals.

---

## 11. Known gaps and roadmap

From `ROADMAP.md` and per-crate `ROADMAP.md` files. Top-priority items:

### P0 — Demo critical path (Act 1 / Act 2 / Act 3)

1. Chat template + EOS stop (so generation doesn't loop). Not started.
2. Token streaming. Not started.
3. ✅ Per-layer FFN format (`layers/`, GPU dispatch) — shipped 2026-04-26; 26B-A4B Metal now at 19.4 tok/s.
4. MoE-aware CPU forward pass. Not started.
5. Wire `RouterIndex` client-side; `POST /v1/expert/{layer}/{expert_id}` server-side — not started.
6. `RemoteExpertBackend` client; reliability pass (timeouts, retries). Not started.

### P0 — Mechanistic surface (lazarus parity)

All eight items M1–M8 shipped: `LayerHook` trait, `RecordHook` / `ZeroAblateHook` / `SteerHook` / `CompositeHook`, activation patching, full logit lens, KV-cache surgery, hooks during multi-token generation, W_E / W_U accessors, PyO3 binding methods.

### P0 — Best-in-class mechanistic interpretability

MI4: golden parity (TRACE final residual matches canonical forward, extend to WalkFfn / patched vindex / Q4K / MoE) — partial. MI5–MI8 (rich attribution objects, causal operators beyond residual replacement, Q4K/MoE trace parity, batched experiment ergonomics) — planned.

### P0 — Interpretability truthfulness + commit semantics

T1–T7 about making the current edit model honest (KNN journal vs compose vs compiled distinction, fixed decomposed TRACE routing, gate-KNN ranking improvements). C1–C3 about explicit COMPILE modes (commit/materialize vs SNAPSHOT) and KNN materialization into FFN edits. Several shipped, several planned.

### P1 — Architecture independence

AI1–AI6 about gating supported families behind executable contracts (extraction, weight writing, forward, trace, prompt rendering), implementing or rejecting MLA architectures, removing scalar attention-geometry fallbacks. Most planned.

### P1 — Research stack promotion

R1–R5 about graduating reusable OV/RD experiment plumbing into `larql-inference::vindex` as stable runtime contracts. R1–R3 shipped.

---

## 12. Possible future directions

Some directions implied by the work but not yet started in earnest:

### 12.1 Native graph traversal (no FFN matmuls at all)

Today's walk-FFN does the gate KNN as a BLAS gemv against a dense matrix. The next step is to recognize that for many inputs, the active set is *predictable* from the residual's "compass direction". With:

- A per-layer **routing graph** (precomputed from training data: residual cluster → likely active-feature set).
- An **HNSW or learned index** over gate vectors at each layer.
- A **template cache** for common prompt prefixes.

…you could turn each FFN layer into a **graph traversal of < K edges** instead of a dense `m`-feature scan. At limit, the forward pass for a familiar prompt reduces to ~constant-time graph walk per layer (a few hundred μs for the whole stack). Several experiments (12, 14, 23, 24) prototype pieces of this.

### 12.2 Attention as routing graph (OV/RD)

Attention is currently the dense holdout: Q/K/V/O are matrix multiplies. The OV/RD experiment classifies attention heads as static (template-fixed), negligible, tableable (precompute as a lookup), addressing-failed, or irreducible. For the static and tableable classes, the QK^T → softmax → V dispatch becomes a graph lookup: "for this prompt template, attention head H always copies position 3 to position 5". The roadmap item R4 promotes the stable parts of this experiment into engine APIs.

### 12.3 Compute embedding (Tier 1, WASM)

`experiments/07_wasm_compute/` shows an FFN slot whose down direction is *not* a vocabulary embedding but a "request for computation" tag. A forward hook intercepts the residual when that tag activates, runs a sandboxed Wasmtime kernel (CP-SAT solver, regex evaluator, arithmetic), and writes the result back as a token-embedding-aligned vector. This breaks the current closed-form "model is the database" thesis open: parts of the graph become **callable computations**, not just lookups.

### 12.4 K reduction (sparse-by-design)

The current K is ~80% of intermediate. Re-quantizing the residual stream's "compass" into a much coarser routing structure (e.g. 1024 routing buckets, 16 active features per bucket) would let `K → 16` per layer. Combined with HNSW it makes the forward pass dominated by the attention block, not the FFN.

### 12.5 Cross-model graph alignment

Two vindexes from related models (Gemma 1B and Gemma 4B) have features at corresponding "concept" positions even though feature indices differ. The cross-model routing experiment (23) has shown that 1B's residual can drive 4B's KNN if you map through an alignment basis. A tooling layer that *automates* alignment would let edits made in a small fast-to-iterate vindex transfer to the production large model — a kind of "knowledge LoRA" delivered as a graph diff.

### 12.6 First-class graph algorithms

`larql-core` already has BFS / shortest path / PageRank scaffolding; few are exposed to LQL. Use cases:

- `BFS FROM "France" RELATION "borders" DEPTH 3` — multi-hop reasoning queries from the graph.
- `PAGERANK ON "country" RELATION "language"` — discover central / canonical entities.
- `SHORTEST_PATH FROM "Mozart" TO "Vienna"` — connection traces.

These are all expressible against the existing edge structure. Today they are research-grade; productizing them would broaden LQL's reach beyond the inference / edit / browse trio.

### 12.7 Quantization for the gate

Gate KNN is bandwidth-bound. The current f16 gate halves vs f32 with no measurable accuracy loss. Q4K gate (with precision-preserving block scales) would quarter again, and an FP4 gate (with the FP8 block-scale hierarchy in `fp4-format-spec.md`) would go further. The key constraint is that MIPS *ranking* must be preserved; the FP4 spec has a Q1 compliance scan with a per-projection fallback to FP8 when ranking would be lost.

### 12.8 Dynamic dispatch routing

Today `WalkFfn` picks one of five paths at construction. A finer policy would pick *per layer* based on:

- Whether overrides are present (heap path, no cache).
- The residual's L2 norm (small residuals → smaller K).
- The previous layer's active-set overlap (skip KNN if same).

The dispatch trait is in place; the policy is pending.

---

## 13. What stands out about the codebase

A few engineering decisions worth highlighting because they aren't obvious from the outside:

- **mmap-everywhere.** Every weight matrix that *can* be mmap'd is. The OS demand-pager is treated as the LRU. Total RSS for a 120B model load is ~1 GB.
- **Hardlinks for COMPILE.** Most COMPILE-INTO-VINDEX bytes are unchanged; LARQL hardlinks them in. APFS / btrfs make this instant.
- **Auto-calibration over magic numbers.** GPU dispatch threshold, FFN sparse-K, gate scale, alpha — most of these have a measurement story behind them and are picked at runtime or empirically validated, not hard-coded.
- **Patches over write-throughs.** The base vindex is *always* readonly. INSERT/DELETE/UPDATE create a patch overlay, never modify base files. This has cascading correctness implications: caches stay valid, multiple users can share a base, edits are revertable.
- **One-way deps.** `model-compute` knows nothing about LARQL. `larql-models` knows nothing about vindex. `larql-vindex` doesn't know about LQL. This stratification is what lets the project ship a `larql` binary, a Python library, an HTTP server, and a Java port (someday) from the same core.
- **Trait-based format dispatch.** Walking sparse FFN is the same code on f32 / Q4K / FP4 / heap-overridden vindexes because the storage decisions live behind one trait. Adding a new format is a single `impl` block.

These are the kinds of choices that make the difference between "interesting research code" and "production-grade, maintainable system". The codebase reads as having gone through several rounds of ruthless audit and consolidation (the substores in `index/core.rs`, the `format/filenames.rs` single-source-of-truth, the parser/executor symmetry).

---

## 14. Summary

LARQL's software architecture follows from one design choice: **the vindex is the model, queried as a graph database**. Storage formats serve graph operations; the inference engine consumes graph operations; the LQL surface exposes graph operations; the COMPILE path turns graph operations back into standard model files. Every other engineering decision (mmap-first, hardlink hardliner, trait-based format dispatch, auto-calibrating backends, patch overlays, slice presets, expert sharding) is in service of that one premise.

The codebase is organized for two consumers:

1. **Research / mech-interp** — wants programmatic forward hooks, residual capture, ablation, steering, activation patching, logit lens, KV surgery — all with zero-cost when not in use.
2. **Production / serving** — wants fast inference (Walk FFN, fused attention, Metal GPU), distributed deployment (slice presets, layer/expert sharding, gRPC grid), and round-trip compatibility with HuggingFace Transformers / GGUF runtimes after edits (COMPILE INTO MODEL).

These are usually separate worlds. LARQL collapses them onto one substrate because the same graph view supports both.
