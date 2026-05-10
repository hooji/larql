# LARQL: Software Architecture

> A guide to the current Rust implementation: what each crate does, how the
> data flows, where the extension points are, and what is incomplete or
> aspirational.

## 1. Workspace at a glance

LARQL is a Cargo workspace at `/home/user/larql`. The dependency graph is
strictly one-way (no cycles), which is enforced by the workspace boundaries:

```
larql-models  ─►  larql-compute  ─►  larql-vindex  ─►  larql-core
                                       │                   │
                                       ├───────────►  larql-inference
                                       │                   │
                                       └───────────►  larql-lql
                                                           │
                                                           ├─►  larql-server  ─►  larql-router
                                                           │
                                                           └─►  larql-cli   (top-level binary)
                                                                    │
                                                                    └─►  larql-python  (PyO3)

                                  model-compute   (portable; no larql-* imports)
                                  larql-router-protocol  (gRPC contract)
                                  kv-cache-benchmark      (standalone)
```

[`Cargo.toml`](../../Cargo.toml) declares all 14 crates. The `model-compute`
crate is intentionally portable and never imports anything `larql-*`; it
will eventually move to a sibling repository while keeping the same name.

## 2. The crate map

### Foundation crates

| Crate | Responsibility |
|-------|----------------|
| `larql-models` | Architecture traits (Gemma/Llama/Qwen/Mistral/etc.), tensor-key naming, weight loaders for safetensors and GGUF, dequant routines (Q4_K, Q6_K, MXFP4) |
| `larql-compute` | Compute-backend dispatch: BLAS (Accelerate on macOS, OpenBLAS on Linux/Windows) and Metal GPU kernels. Auto-calibration picks CPU vs GPU per matmul size at startup. |
| `model-compute` | Bounded native compute primitives (arithmetic/datetime kernels) plus an optional WASM runtime (wasmtime). Used at *compile* time to resolve expressions like `sum(1..100)`. Has zero `larql-*` dependencies. |

### Vindex and graph crates

| Crate | Responsibility |
|-------|----------------|
| `larql-vindex` | The vindex lifecycle: streaming extraction (mmap; never loads the full model), KNN over gate vectors, zero-copy mmap loading, split weight files, readonly base + patch overlay, clustering, f16 storage, Vindexfile parser. The flagship crate. |
| `larql-core` | Pure graph algorithms over the extracted edge graph: BFS, PageRank, shortest-path, merge, diff. Independent of the vindex storage layout. |

### Inference and language crates

| Crate | Responsibility |
|-------|----------------|
| `larql-inference` | Forward pass — BLAS-fused attention, walk-FFN, Metal GPU path, mechanistic-interp hook trait, multi-token generation, KV cache surgery |
| `larql-lql` | Lexer, parser, AST, executor, REPL. The five sub-modules (`lifecycle`, `query`, `mutation`, `introspection`, `trace`) appear symmetrically in both `parser/` and `executor/`. |

### Service and tool crates

| Crate | Responsibility |
|-------|----------------|
| `larql-server` | Axum HTTP + tonic gRPC server exposing vindexes over the network. Supports FFN-only mode, MoE expert sharding, layer sharding |
| `larql-router` | Layer-range router for distributed FFN; static route table |
| `larql-router-protocol` | Generated gRPC stubs |
| `larql-experts` | (workspace member; minimal at present) |
| `larql-cli` | Top-level `larql` binary; thin dispatcher into `commands/extraction/` and `commands/query/` |
| `larql-python` | PyO3 bindings; built with maturin under uv. Module name is `larql._native`. |

### Standalone / benchmark

| Crate | Responsibility |
|-------|----------------|
| `kv-cache-benchmark` | Comparison harness for four KV-cache engines: MarkovRS, UnlimitedContext, TurboQuant, Apollo |

## 3. Core data structures

The two structs that matter most for understanding the system are
`VectorIndex` and `PatchedVindex`.

### 3.1 `VectorIndex`

Defined in
[`crates/larql-vindex/src/index/core.rs`](../../crates/larql-vindex/src/index/core.rs).
After the 2026-04-25 refactor it is divided into four substores:

```rust
pub struct VectorIndex {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub layer_range: Option<(usize, usize)>,   // for sharding

    pub gate:        GateStore,        // gate matrix mmap/heap, HNSW index, decode caches
    pub ffn:         FfnStore,         // Q4_K / FP4 FFN weights, dequant cache
    pub projections: ProjectionStore,  // lm_head, attention OV/QK projections (mmap)
    pub metadata:    MetadataStore,    // FeatureMeta per slot, down-vector overrides
}
```

Each substore can operate in two modes:

- **Heap** mode (used during build): `Option<Array2<f32>>` per layer — easy to mutate.
- **Mmap** mode (production load): `Arc<Mmap>` over the on-disk file, sliced per layer. Zero-copy.

The flat layout means a Gemma-3-4B vindex resident set is the size of the
*current working set* (typically a few hundred MB even for a 16 GB model),
not the full model file size.

### 3.2 `PatchedVindex`

Defined in
[`crates/larql-vindex/src/patch/overlay.rs`](../../crates/larql-vindex/src/patch/overlay.rs).
This is the read-only overlay that all mutations flow through:

```rust
pub struct PatchedVindex {
    pub base: VectorIndex,                                // immutable on-disk
    pub patches: Vec<VindexPatch>,                        // applied patches in order
    pub overrides_meta: HashMap<(usize, usize), Option<FeatureMeta>>,
    pub overrides_gate: HashMap<(usize, usize), Vec<f32>>,
    pub deleted: HashSet<(usize, usize)>,
    pub knn_store: KnnStore,                              // L0 residual-key KNN (Architecture B)
}
```

A subtle but important asymmetry: **gate-vector overrides live in the
overlay** (`overrides_gate`), while **down-vector overrides are forwarded
to `base.metadata.down_overrides`** so the FFN walk can pick them up
without reflating the base activation. This separation is exactly what
the constellation-insert paper requires (so that small `α` per-layer
contributions accumulate cleanly without blowing up the residual at any
single layer).

### 3.3 `FeatureMeta`

Defined in `crates/larql-vindex/src/index/types.rs`. Per-feature metadata:

```rust
pub struct FeatureMeta {
    pub top_token_id: u32,        // primary output of this feature
    pub c_score: f32,             // confidence 0..1
    pub top_k: Vec<TopKEntry>,    // alternatives
    pub label: Option<String>,    // probe-confirmed label
    pub confidence: f32,
    pub selectivity: f32,
}
```

Stored on disk in `down_meta.bin` (binary, ~80× smaller than JSON).

## 4. The vindex on-disk format

[`docs/format.md`](../format.md) documents the layout in detail.
Summary of the directory:

```
gemma3-4b.vindex/
  index.json              VindexConfig: extract_level, layer_bands, layers[].offset, fp4
  weight_manifest.json    tensor-key -> (offset, length) table
  tokenizer.json          HuggingFace tokenizer
  relation_clusters.json  discovered relation clusters (post-hoc)
  feature_labels.json     probe-confirmed labels

  gate_vectors.bin        layer-concatenated [intermediate_size, hidden] f32/f16
  embeddings.bin          [vocab_size, hidden] embedding matrix
  down_meta.bin           DMET binary header + per-feature top-k metadata

  attn_weights.bin        Q,K,V,O per layer (only at level >= attention)
  norms.bin               LayerNorm parameters (only at level >= attention)
  up_weights.bin          [intermediate_size, hidden] per layer (only at level >= inference)
  down_weights.bin        [hidden, intermediate_size] per layer (only at level >= inference)
  lm_head.bin             output projection (only at level = all)

  attn_weights_q4k.bin    quantised path (when --quant q4k)
  interleaved_q4k.bin     interleaved Q4_K FFN gate/up + Q6_K (or Q4_K) down

  gate_vectors_fp4.bin    optional FP4 / FP8 block storage (exp 26)
  up_features_fp4.bin
  down_features_fp8.bin
  fp4_compliance.json

  layers/layer_NN.weights per-layer FFN format for MoE
```

The vindex is a *directory*, not a single file, so individual components
can be hard-linked, replaced, or sliced without rewriting the whole
thing.

### Three extraction levels

| Level | Includes | Size (Gemma 3-4B f16) | LQL operations enabled |
|-------|----------|----------------------|------------------------|
| `browse`     | gates + embeddings + down_meta + tokenizer | ~3 GB  | DESCRIBE, WALK, SELECT |
| `inference`  | + attention + norms + up + down            | ~6 GB  | + INFER, EXPLAIN INFER, TRACE |
| `all`        | + lm_head + compile metadata               | ~10 GB | + COMPILE INTO MODEL |

There are also intermediate "slice" presets (`attn`, `embed`, `client`,
`server`, `expert-server`, `router`, `browse`, `all`) used by
`larql slice` to carve a vindex into deployment-ready pieces (e.g.
laptop holds attention + embeddings, a remote CPU box holds the FFN
and serves it over HTTP/gRPC).

## 5. Core operations and where they live

### 5.1 Gate KNN (the heart of the walk)

Implemented in
[`crates/larql-vindex/src/index/compute/gate_knn/`](../../crates/larql-vindex/src/index/compute/).
Three paths:

- `gate_knn_mmap_fast` — direct gemv over the f32 gate mmap, top-K via
  partial heap. Hot path on production loads.
- `gate_walk` — batched BLAS gemm across multiple residual queries,
  used by inference engine for parallel position scans.
- Top-K extractors `top_k_from_scores`, `top_k_by_abs` for selecting by
  signed value or absolute magnitude (the latter preserves
  inverter/suppressor features).

Latency on Gemma-3-4B: **~2.78 ms per layer** for production-dim
KNN (intermediate=10240, hidden=2560).

### 5.2 Walk FFN

Implemented in
[`crates/larql-inference/src/vindex/walk_ffn/exact.rs`](../../crates/larql-inference/src/vindex/walk_ffn/).
The pipeline per layer:

1. Compute `gate = W_gate · x` and `up = W_up · x` from safetensors-loaded weights.
2. Apply `SiLU(gate) ⊙ up` to get per-feature activations.
3. Gate-KNN: pick top-K features.
4. Sparse down projection: read `K` columns of the down matrix from the
   feature-major mmap'd file (`down_features.bin`), gemm them against the
   activation slice, sum into output.

There is also `sparse_ffn_forward_with_overrides` which substitutes
override down vectors from `MetadataStore.down_overrides` for any
feature that has been INSERTed.

### 5.3 `install_edge` (the compile primitive)

Implemented in
[`crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs`](../../crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs).
Lines 53–124. Takes a slot, a normalised trigger, a normalised write,
a `gate_scale` (typically 30.0) and an `alpha_mul`. Reads reference norms
from the original slot, then writes:

```rust
gate[s, :]  ←  trigger * (g_norm * gate_scale / ‖trigger‖)
up[s,   :]  ←  trigger * (u_norm / ‖trigger‖)
down[:, s]  ←  write * (d_norm / ‖write‖) * alpha_mul
```

This is the lowest-level step of the COMPILE verb. It is currently a
single-call-site function inside `larql-cli`; the AGENTS.md notes that
when a second consumer needs it (e.g. a future TinyModel), it will be
extracted into its own crate.

### 5.4 The patch system

[`crates/larql-vindex/src/patch/`](../../crates/larql-vindex/src/patch/)
defines `VindexPatch` (a serialised JSON file `.vlp`) and `PatchOp`:

```rust
pub enum PatchOp {
    Insert     { layer, feature, gate, up, down, meta },
    InsertKnn  { layer, entity, target, target_id, key },
    Delete     { layer, feature, reason },
    DeleteKnn  { layer, entity },
    Update     { layer, feature, gate?, up?, down?, meta? },
}
```

Patches stack in application order. Later patches override earlier ones
for the same `(layer, feature)`. Removing a patch is instantaneous (it
just unsets the overlay entries). The base `gate_vectors.bin`,
`down_weights.bin`, etc. are *never* mutated.

### 5.5 COMPILE flow

Two destinations:

- **`COMPILE CURRENT INTO VINDEX <path>`** — produces a fresh standalone
  vindex by hardlinking unchanged base files (instant on APFS) and
  rewriting only `down_weights.bin` column-wise where overrides exist.
  The compiled vindex needs no overlay logic at load time.

- **`COMPILE CURRENT INTO MODEL <path> FORMAT safetensors`** — calls
  `install_edge` for every accumulated insert, writes new safetensors
  (or GGUF) containing the dense gate/up/down matrices with the
  inserted edges materialised. Output runs in any external inference
  engine (HuggingFace Transformers, llama.cpp, MLX) without special
  loaders.

## 6. The LQL surface

Defined in
[`crates/larql-lql/src/ast.rs`](../../crates/larql-lql/src/ast.rs).
Statement categories:

| Category | Statements | Purpose |
|----------|-----------|---------|
| Lifecycle | `EXTRACT`, `COMPILE`, `DIFF`, `USE` | model management |
| Browse | `WALK`, `DESCRIBE`, `SELECT`, `EXPLAIN WALK` | query (no forward pass) |
| Inference | `INFER`, `EXPLAIN INFER` | full forward pass |
| Trace | `TRACE … FOR …`, `TRACE … DECOMPOSE …`, `TRACE … SAVE …` | residual-stream decomposition |
| Mutation | `INSERT`, `DELETE`, `UPDATE`, `MERGE`, `REBALANCE` | edit knowledge |
| Patch | `BEGIN PATCH`, `SAVE PATCH`, `APPLY PATCH`, `REMOVE PATCH`, `SHOW PATCHES`, `DIFF … INTO PATCH` | persistent overlays |
| Introspection | `SHOW {RELATIONS, LAYERS, FEATURES, ENTITIES, MODELS, PATCHES}`, `STATS` | catalog queries |
| Pipe | `\|>` | chain statement results |

Parser modules (`parser/lifecycle.rs`, `parser/query.rs`,
`parser/mutation.rs`, `parser/introspection.rs`, `parser/trace.rs`) are
mirrored by executor modules of the same names. To add a new statement
you touch the AST, then both sides.

INSERT supports two modes:

- **`MODE KNN`** (default) — stores a residual key in `KnnStore` so
  retrieval picks up the new fact at inference time. Scales to ~25K
  edges. No weight mutation.
- **`MODE COMPOSE`** — calls a `install_compiled_slot` path that writes
  gate/up/down slots so the inserted feature participates in the
  forward pass directly. Capped at ~5–10 facts per layer because slot
  pressure increases collateral degradation.

The default behaviour (`INSERT INTO EDGES …` with no extra clauses)
runs the validated multi-layer constellation (~8 layers × `α`=0.25)
described in §5 of the math doc.

## 7. Inference engine

[`crates/larql-inference/src/lib.rs`](../../crates/larql-inference/src/lib.rs)
wires together attention, FFN, and the hook system.

### 7.1 BLAS-fused attention

Per query position, attention is computed without ever materialising the
`[seq, seq]` attention matrix:

1. `scores[0..=qi] = K[0..=qi] · Q[qi]` (BLAS gemv)
2. two-pass softmax (max, exp, normalise)
3. `output = V[0..=qi]^⊤ · softmax_scores` (BLAS gemv)

Temporary buffer is `O(seq)` per position, not `O(seq²)`. Measured
1.6× faster than a materialised attention path on M-series; 10× memory
savings at seq=512.

### 7.2 Hook system

The forward pass exposes five interception points:

```
pre_layer
  → on_pre_layer(layer, &h)
attention
  → on_attention_weights(layer, &w)        [capture]
  → on_post_attention(layer, &mut h)       [intervention]
FFN
  → on_ffn_activation(layer, &gate)        [capture]
post-layer
  → on_post_layer(layer, &mut h)           [intervention]
```

Built-in hooks:

| Hook | Purpose |
|------|---------|
| `RecordHook` | snapshot residuals at chosen layers |
| `ZeroAblateHook` | zero a feature/head |
| `SteerHook` | add `α·v` mid-forward |
| `CompositeHook` | chain multiple hooks |

Forward path is **zero-cost when no hook is registered** — the trait
dispatch compiles to a no-op. The Python side (`larql._native.WalkModel`)
exposes all the same primitives via PyO3.

### 7.3 Backends

The walk-FFN can run against:

- `LocalWalkBackend` — local mmap'd vindex (default).
- `RemoteWalkBackend` — POST `/v1/walk-ffn` to a remote `larql serve`.
- `RemoteMoeBackend` — gRPC `ExpertService` for MoE expert dispatch.

This is what powers the "laptop runs attention, CPU farm runs FFN"
deployment topology and the MoE expert-grid.

## 8. CLI architecture

[`crates/larql-cli/src/main.rs`](../../crates/larql-cli/src/main.rs)
is a thin dispatcher. Every subcommand lives in
`commands/extraction/` or `commands/query/` and is wired into the
top-level `Commands` enum.

Top-level verbs:

| Verb | Purpose |
|------|---------|
| `extract` (alias `extract-index`) | decompile a model into a vindex |
| `convert` | GGUF → vindex |
| `compile` | recompile vindex back to safetensors / GGUF / a fresh vindex |
| `slice` | carve a vindex into deployment slices |
| `pull`, `publish`, `link`, `list`, `rm` | HuggingFace-style cache management |
| `serve` | start HTTP + gRPC server |
| `run`, `chat` | one-shot or interactive inference |
| `repl`, `lql` | LQL REPL or one-liner |
| `bench` | benchmark suite |
| `dev <subcmd>` | research/interpretability tools (20+ sub-commands) |

`larql walk …` etc. exist as backwards-compatible aliases that
trampoline into `larql dev walk …`.

## 9. Server, sharding, and the grid

[`crates/larql-server`](../../crates/larql-server) ships an Axum
HTTP server plus a tonic gRPC service. Endpoints include:

- `POST /v1/walk-ffn` (CPU FFN walk over residual)
- `POST /v1/walk-ffn-q8k` (Q8K wire format, dequant on the receiver)
- `POST /v1/expert/batch` (MoE expert dispatch)
- `GET  /v1/stats.q4k_ffn` (cache resident set, decoded buffer status)

Sharding axes:

- **By layer**: `--layers 0-19` makes a server hold only those layer slices.
- **By expert (MoE)**: `--experts 0-63`.
- **2-D grid**: a layer-shard server fans out to expert servers via
  `--moe-shards`.
- **Embed split**: a 3-tier client + embed-server + FFN-server topology
  reduces the laptop side to ~310 MB on a 4 B model.

`larql-router` provides a static layer-range router that the client can
target with a single `--ffn http://router:9090`. There's a deployed
demo on fly.io serving Gemma 4 26B-A4B from CPU-only servers — the FFN
is just memory-mapped data; no GPU is required.

## 10. Build, test, run

[`Makefile`](../../Makefile) provides the canonical entry points:

```
make ci             # fmt-check + clippy -D warnings + tests
make fmt            # cargo fmt --all
make lint           # clippy -D warnings
make test           # workspace tests (no model-backed ignored ones)
make test-full      # + integration tests
make test-models    # + golden-output tests on real Gemma 3/4 models
make bench          # criterion benchmarks
make python-build   # uv + maturin develop --release
make python-test    # uv run pytest
```

Reported scale (per ROADMAP.md, 2026-05-02): **2,000+ tests**, zero
build warnings, Gemma 3-4B Metal at 83–84 tok/s (vs Ollama 99) on
M3 Max.

## 11. What exists vs. what's planned

### Solid and shipping
- Three-level extraction (browse / inference / all)
- Gate KNN, walk FFN, BLAS-fused attention, Metal GPU dispatch
- Patch overlay, COMPILE INTO VINDEX, COMPILE INTO MODEL (safetensors)
- LQL parser + executor with all 20+ statement types
- Mechanistic-interp hooks (capture / ablate / steer / patch / lens)
- Python bindings, REPL, CLI, HTTP+gRPC server
- Single-machine MoE grid (validated 2-shard)
- HuggingFace publish/pull with collections and slice awareness

### In-progress (per `ROADMAP.md`)
- Chat-template handling and EOS-stop in generation (blocking Act 1 demo)
- Token streaming (blocking Act 1 demo)
- Per-layer FFN format Phase 2 GPU dispatch (Phase 1 shipped)
- MoE-aware CPU forward pass

### Planned / aspirational
- Architecture independence hardening (gated FFN contracts, MLA, attention-geometry allocation)
- Truthful KNN overlay tags, TRACE evidence pinning, WalkFfn patching
- Acceptance test harness for COMPILE round-trip
- Token-level attribution in TRACE (currently logit-level)
- Causal operators beyond residual replacement
- Static grid scaling to 1T-parameter (Kimi K2.6 / DeepSeek V4-class) deployment

## 12. Possible directions for future development

The architecture already supports these naturally; they are listed as
extensions, not rewrites:

1. **`install_edge` extraction.** Move the primitive into its own
   `larql-edge` crate so it can be reused by TinyModel and any
   external compiler.
2. **Higher-order edges.** Generalise PatchOp::Insert to support
   multi-input or multi-output features. The dense weight write is
   already low-rank-friendly; a rank-2 insert would fit.
3. **Manifold-compressed gates.** Experiment 02 found 99% variance in
   ~15 dimensions across knowledge-band gates. Storing gate vectors
   in a learned 15-D basis would reduce the gate file by ~150×.
4. **Attention-graph extraction.** Experiments 16, 18, 19, 24, 38 are
   converging on an attention-side equivalent of the FFN graph. A
   `attn_index.bin` storing per-head OV/QK low-rank decompositions
   would make attention also queryable.
5. **Dark-space probing.** A scheduled job that periodically probes
   the 85% "dark" features for emergent labels (cluster centroids,
   relation patterns, syntax circuits) would close the structural-vs-factual
   distinction.
6. **Pluggable confidence models.** The current `c = c_in · c_out / max`
   is a heuristic. A learned confidence model could be slotted in as
   a `ConfidenceProvider` trait and selected at vindex build time.
7. **Distributed COMPILE.** Compiling a 31 B model with 1,000 patches
   should parallelise trivially across slot writes; the column-wise
   `down_weights.bin` rewrite is embarrassingly parallel.
8. **WASM expert kernels.** `model-compute` already supports wasmtime;
   shipping a "WASM expert" registry would let users plug in custom
   feature behaviours that compose with the FFN graph.
9. **Inverse-problem solve for COMPOSE inserts.** REBALANCE currently
   does a fixed-point iteration. A direct ridge-regression solve
   (essentially the MEMIT decomposition, already prototyped in
   `experiments/`) could replace the iteration with a closed-form
   step.
10. **Architecture-agnostic frontends.** Add explicit ungated-FFN
    (GPT-2-style) and untied-LM-head paths so the same toolchain
    works on the older transformer family without special-casing.

## 13. Files worth knowing

| Topic | File |
|-------|------|
| Workspace | [`Cargo.toml`](../../Cargo.toml) |
| Build/test | [`Makefile`](../../Makefile) |
| Roadmap | [`ROADMAP.md`](../../ROADMAP.md) |
| Format spec | [`docs/format.md`](../format.md) |
| AGENTS overview | [`AGENTS.md`](../../AGENTS.md) |
| `VectorIndex` | [`crates/larql-vindex/src/index/core.rs`](../../crates/larql-vindex/src/index/core.rs) |
| `PatchedVindex` | [`crates/larql-vindex/src/patch/overlay.rs`](../../crates/larql-vindex/src/patch/overlay.rs) |
| `install_edge` | [`crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs`](../../crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs) |
| LQL AST | [`crates/larql-lql/src/ast.rs`](../../crates/larql-lql/src/ast.rs) |
| Walk FFN | [`crates/larql-inference/src/vindex/walk_ffn/`](../../crates/larql-inference/src/vindex/) |
| Hook trait | [`crates/larql-inference/src/forward/`](../../crates/larql-inference/src/) |
| Server | [`crates/larql-server/src/`](../../crates/larql-server/src/) |
| CLI main | [`crates/larql-cli/src/main.rs`](../../crates/larql-cli/src/main.rs) |
