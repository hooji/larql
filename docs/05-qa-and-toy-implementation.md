# LARQL — Questions and a Toy-Implementation Sketch

> Direct answers to three framing questions plus a sketch of a Java toy version that operates on nanochat / GPT-2.

---

## Q1. Is the graph-network view a more direct understanding of what LLMs actually do? Could the matrix multiplications be replaced by graph traversal?

**Short answer: yes to the first half, mostly no (today) to the second half.** The graph view is mathematically more direct *for the FFN*. The performance argument for replacing matmul with graph traversal is real but currently only paying off in narrow regimes. Here is the long answer.

### Mathematically, the graph view is the direct one

The standard form of a gated FFN is:

```
δ_ffn(x) = W_down · ( σ(W_gate · x) ⊙ (W_up · x) )
```

This is "do three matmuls and a Hadamard". It is a notational compaction. The literal computation, when you expand the matmul as a sum of outer products, is:

```
δ_ffn(x) = ∑_i d_i · σ(g_iᵀ x) · (u_iᵀ x)
```

That is "for each of the m features, decide if it fires (the σ(g_iᵀ x) factor), measure how strongly (multiplied by u_iᵀ x), then add its output direction d_i to the residual". **The matmul form is a vectorised expression of a graph operation.** The graph operation is the primary thing; the matmul is the BLAS-friendly notation.

This isn't a controversial claim — it's literally what the equation says. What's surprising is how rarely anyone has built tooling that takes the graph form seriously. LARQL's central contribution is to write down storage formats, query languages, and edit operations that treat each (g_i, d_i) as a first-class object.

The walk-FFN boundary sweep (`docs/walk-boundary-sweep.md`) confirms the equivalence empirically: replace the dense matmul with the explicit per-feature graph sum at every layer cut from L0 to L34 on Gemma 3 4B; top-1 token and probability are identical at every boundary. **Same answer, different shape of computation.**

### So why hasn't matmul been replaced already?

Because BLAS is *very* good at what it does. A single `cblas_sgemm` call on a 10240×2560 matrix runs at 2–4 TFLOPS on AMX, which means the dense matmul takes ~6 ms per layer on Gemma 3 4B. A naive per-feature loop would be 10240 separate gemv calls — orders of magnitude slower because of dispatch and cache overhead.

The walk-FFN, when it pays off, pays off because:

- The down read becomes a sequential mmap stream from a feature-major file rather than a strided gather from a safetensors file. The OS page cache becomes the working set.
- For sparse top-K (K << m), the up + down work is genuinely smaller — but you still need to compute *all* gate dot products to find the active set. A BLAS gemv against the whole gate matrix is the cheapest way to do that.

So at K=8092 of m=10240 (where LARQL operates today on Gemma), the walk and the dense matmul are within 4% of each other (517 ms vs 535 ms). The walk is *slightly* faster, but it is not the dramatic order-of-magnitude win the graph view might suggest.

**Where it would dramatically pay off:**

- **Truly sparse activations.** If you trained a model where only K=64 of m=10240 features fire on average per token (Switch Transformer style, or learned TopK), the walk skips computing 99.4% of the up/down work. Today's models aren't trained for this; their gate distributions are diffuse.
- **Bandwidth-bound regime.** If FFN weights are at Q4_K (4 bits/weight = 1.25 GB instead of 5 GB at f16), the dense matmul becomes bandwidth-bound. The walk doesn't help unless K is also small. Both are required for the win.
- **Predictable active sets.** If you can predict the active set without computing all gate dots — via clustering the residual into "compass directions" and storing a precomputed active set per cluster — you skip the gate gemv entirely. This is the direction of experiments 23/24 (routing geometry) and is genuinely promising but not in production.
- **Repeated identical residuals.** Paraphrase collapse: residuals from semantically equivalent prompts have cos > 0.99, and the active set is identical. The L1/L2 FFN activation cache (`docs/ffn-cache.md`) hashes the active feature set as a cache key; cache hits skip the per-feature compute entirely. Today this provides 60–80% hit rates on common factual queries.

### A more honest framing

The graph view is the **conceptually correct** view of what FFN features do. The matmul is the **operationally correct** view of how to multiply numbers fast. They are the same computation, viewed at different granularity.

The win from the graph view today is not so much "replacing matmuls with traversal" as it is:

- **Inspectability.** You can name and label features, browse them, and identify circuit types because you store them as discrete objects.
- **Editability.** You can write a new feature to slot `i` because slot `i` is addressable.
- **Distributability.** You can shard features across machines (MoE expert sharding) because they're discrete things to shard.
- **Mmap-friendliness.** Feature-major storage lets the OS demand-page just the rows you touch.

The performance win is a happy bonus that arrives in specific regimes (Q4_K + small K, paraphrase cache hits, MoE expert sparsity). The conceptual win — that the model is now a thing you can read, edit, and reason about as a graph — is the bigger deal.

Looking forward, the path to a *truly* graph-traversal-only forward pass requires three things to mature:

1. Models trained for sparse activation (intentionally, not as a side effect).
2. Compact, bandwidth-friendly storage (FP4 / FP8 / Q4K with preserved ranking).
3. Approximate / learned indexes (HNSW + clustering + template caches) so the gate KNN itself becomes sub-linear.

LARQL has prototypes of all three, but none are yet at the point where they replace dense matmul as the default. The infrastructure for them to graduate is in place.

---

## Q2. Can the work be boiled down to something simple enough for a relatively compact implementation?

**Yes — the *core* is small.** Here is what an MVP needs, at the conceptual level. Production LARQL adds many engineering layers on top (Q4K, FP4, MoE, distributed serving, hooks, trace storage, patches, COMPILE, …) but you can run the central demonstrations of "extract a vindex, query it as a graph, walk inference through it, edit a fact, see the edit reflected in inference output" with a much smaller surface.

### What the MVP must do

Given a HuggingFace transformer model with gated FFN (Llama / Mistral / Qwen / Gemma):

1. **Extract a vindex.** Read the safetensors files, copy `W_gate` rows into a flat float file, copy embeddings into another flat file, write a `index.json` describing layer offsets. (A few hundred lines.)
2. **Gate KNN.** Given a residual `x` and a layer `L`, compute `gate_matrix[L] @ x` and return the top-K indices by absolute value. (~30 lines.)
3. **Walk forward pass.** Embed the prompt → for each layer: attention block, post-attention RMS norm, gate KNN, sparse FFN aggregate, residual add → final norm + lm_head + softmax. (Most of this is just transformer code; the FFN piece is short.)
4. **DESCRIBE.** Embed an entity, gate KNN at each knowledge-band layer, lookup top output token per active feature via embedding-projection of the down vector. (~50 lines.)
5. **INSERT (constellation method).** Run a forward pass on a synthesized prompt, capture residuals at chosen layers, write `(gate, up, down)` for an unused slot at each layer using the published norms convention. (~100 lines.)
6. **COMPILE.** Read the model's safetensors, splice the override columns into `down_weights`, write fresh safetensors. (~100 lines.)

That's the headline demo. Production LARQL has 600+ tests in `larql-vindex` alone, but the *conceptual* core is much smaller than the full system suggests. As a rough estimate: **a Python implementation in 1,500–2,000 lines** could reproduce the headline behaviours on a small dense model. (Production LARQL is around 50,000+ lines of Rust, but the bulk of that is the production-grade engineering: Q4K, Metal, MoE, sharding, FP4, mech-interp hooks, gRPC, REPL UX, HuggingFace publish.)

### Minimal pseudocode

```python
# ─── Extract ──────────────────────────────────────────────────────────────
def extract_vindex(model_dir, output_dir):
    cfg = read_config(model_dir)
    arch = detect_arch(cfg)              # Llama / Mistral / Gemma / etc.

    out = {}
    out["index.json"] = {
        "num_layers": cfg.n_layers, "hidden": cfg.hidden,
        "intermediate": cfg.intermediate, "vocab": cfg.vocab,
        "embed_scale": arch.embed_scale, "model": cfg.name,
    }

    embeddings = read_tensor(model_dir, arch.embed_key())     # [vocab, hidden]
    write_bin(output_dir / "embeddings.bin", embeddings.astype("float16"))

    gate = []
    for L in range(cfg.n_layers):
        g_L = read_tensor(model_dir, arch.gate_key(L))         # [intermediate, hidden]
        gate.append(g_L)
    write_bin(output_dir / "gate_vectors.bin",
              concatenate(gate).astype("float16"))

    # down_meta: top-10 vocabulary tokens each feature points at.
    down_meta = []
    for L in range(cfg.n_layers):
        d_L = read_tensor(model_dir, arch.down_key(L))         # [hidden, intermediate]
        for i in range(cfg.intermediate):
            scores = embeddings @ d_L[:, i]                    # [vocab]
            top_k = argpartition(-scores, 10)[:10]
            down_meta.append({"layer": L, "feature": i,
                              "top_k": top_k.tolist(),
                              "scores": scores[top_k].tolist()})
    write_json(output_dir / "down_meta.json", down_meta)

    # For inference / compile: also write attn / norms / up / down / lm_head.
    ...

# ─── Gate KNN ────────────────────────────────────────────────────────────
def gate_knn(vindex, layer, residual, top_k=8092):
    g_L = vindex.gate_layer(layer)        # mmap'd [intermediate, hidden] f16
    scores = g_L @ residual               # [intermediate]
    return top_K_by_abs(scores, top_k)    # [(feat_idx, score)]

# ─── Sparse FFN walk ─────────────────────────────────────────────────────
def walk_ffn(vindex, weights, layer, x):
    # x: [seq, hidden] post-attention residual
    out = zeros_like(x)
    for s in range(x.shape[0]):
        x_s = x[s]
        hits = gate_knn(vindex, layer, x_s)
        for (i, gate_score) in hits:
            up_score = weights.up_row(layer, i) @ x_s
            a_i = silu(gate_score) * up_score
            out[s] += a_i * weights.down_col(layer, i)
    return out

# ─── DESCRIBE ────────────────────────────────────────────────────────────
def describe(vindex, entity, knowledge_layers):
    ev = embed(vindex, entity)                  # [hidden]
    edges = []
    for L in knowledge_layers:
        for (i, score) in gate_knn(vindex, L, ev, top_k=10):
            meta = vindex.feature_meta(L, i)    # which tokens its down points at
            edges.append((meta["top_token"], L, i, score))
    return aggregate_by_target(edges)

# ─── INSERT (constellation method) ───────────────────────────────────────
def insert(vindex, weights, entity, relation, target,
           install_layers=range(20, 28), alpha=0.25, gate_scale=30.0):
    prompt = f"The {relation} of {entity} is"
    residuals_per_layer = forward_capture(weights, prompt, install_layers)
    target_embed = embed(vindex, target)
    target_unit = target_embed / norm(target_embed)
    for L in install_layers:
        slot = find_free_slot(vindex, L)
        r = residuals_per_layer[L]
        r_unit = r / norm(r)
        g_norm, u_norm, d_norm = layer_typical_norms(weights, L)
        vindex.set_gate(L, slot, r_unit * g_norm * gate_scale)
        vindex.set_up  (L, slot, r_unit * u_norm)
        vindex.set_down(L, slot, target_unit * d_norm * alpha)

# ─── COMPILE INTO MODEL ──────────────────────────────────────────────────
def compile_to_safetensors(vindex, weights, output_dir):
    for L in vindex.layers:
        for slot, override in vindex.down_overrides(L):
            weights.down_proj[L][:, slot] = override
        for slot, override in vindex.up_overrides(L):
            weights.up_proj[L][slot, :] = override
        # gate is intentionally NOT written back — see math foundations §7.1
    write_safetensors(output_dir, weights)
```

Maybe ~700 lines of Python with reasonable error handling and commenting, plus a stripped-down forward pass (~500 lines for attention + norms + sampling), plus extraction (~300 lines), plus a small CLI (~200 lines). **~1,700 lines for a working toy.** No CUDA, no Triton, no ndarray gymnastics; numpy + safetensors + tokenizers is enough.

### What you don't need for a toy

- **GPU acceleration.** A 117M GPT-2 is fast enough on CPU.
- **Q4K / Q6K / FP4.** Use f16 or f32 storage.
- **MoE.** GPT-2 / nanochat / Gemma 3 4B are dense.
- **Distributed serving.** Single process is fine.
- **Patches as a separate file format.** You can keep overrides in memory.
- **Mech-interp hooks.** A simple `forward_capture(weights, prompt, layers)` is enough.
- **HuggingFace publish flow.** Local files are fine.
- **The full LQL parser.** A handful of CLI subcommands (`extract`, `describe`, `insert`, `infer`, `compile`) cover the demo.

The minimum viable demonstration is:

```bash
$ toy extract gpt2-small.safetensors -o gpt2.vindex
$ toy describe gpt2.vindex "Paris"
  L9:F2842  → "France"   12.4
  L10:F1573 → "city"     11.9
  ...
$ toy infer gpt2.vindex "The capital of France is"
  Paris (47%), London (12%), ...
$ toy insert gpt2.vindex "Atlantis" "capital-of" "Poseidon"
  Inserted across L8-L10, alpha=0.25
$ toy infer gpt2.vindex "The capital of Atlantis is"
  Poseidon (62%), ...
$ toy compile gpt2.vindex -o gpt2-edited.safetensors
$ python -c "from transformers import GPT2LMHeadModel; \
             m = GPT2LMHeadModel.from_pretrained('gpt2-edited.safetensors'); \
             print(m.generate('The capital of Atlantis is'))"
```

Six commands. Round-trip. Works on a laptop CPU.

---

## Q3. How hard is a "toy" version on GPT-2 / nanochat? Could it be done in Java? Sketch the architecture.

### How hard

**Quite tractable.** GPT-2 is the easiest possible substrate:

- **Small.** 117M params for `gpt2-small`, 345M for `gpt2-medium`. Fits comfortably in laptop RAM.
- **Dense.** No MoE, no expert routing.
- **Unfused.** Standard attention, GELU FFN — no SwiGLU, no GQA, no RoPE, no per-layer-embedding, no logit softcap.
- **Open weights.** Available in safetensors on HuggingFace, well-understood layout.
- **Tied embeddings.** `lm_head = embed.T` simplifies the lens / unembedding.

`nanochat` (Karpathy) is even simpler — explicitly designed as a teaching scaffold, ~2,000 lines of Python. The model architecture is essentially a simplified GPT.

The conceptual delta from production LARQL to a toy is small enough that the toy is a **weekend project for someone familiar with transformer internals**. The hardest part is getting the small numerical details right (RMS-norm vs LayerNorm, GELU exact / tanh approximation, byte-pair vs SentencePiece tokenizers, `pre-norm` vs `post-norm` order). The graph operations themselves are short and easy to test.

For **GPT-2 specifically** there are two minor adjustments because it uses dense FFN, not gated:

```
GPT-2 FFN:    out = W_out · GELU(W_in · x + b_in) + b_out
Gated FFN:    out = W_down · ( σ(W_gate · x) ⊙ (W_up · x) )
```

So in the toy:

- **Gate = W_in rows.** Each row is "what direction this feature responds to". Same role as gate vector.
- **No up vector.** The activation is `σ(g_iᵀ x)` directly, no `(u_iᵀ x)` multiplier. (You can pretend `u_i = unit_vector` and `(u_iᵀ x) = constant`, but it's cleaner to special-case dense FFN.)
- **Down = W_out columns.** Same as before.
- **Constellation insert** — `g_i ← unit(residual) × g_norm`, `d_i ← unit(target_embed) × d_norm × α`. No `u_i`.

The procedure works the same way; you just don't carry an up vector around.

### Could it be done in Java?

**Yes, comfortably.** Java is in fact a reasonable choice for this kind of system because:

- **Memory mapping is first-class.** `FileChannel.map(MapMode.READ_ONLY, …)` returns a `MappedByteBuffer` that the OS demand-pages just like Rust's `Mmap`. The vindex `mmap`-everywhere strategy ports directly. For files > 2 GB you use `MemorySegment.mapFile(...)` from the new FFM API (`java.lang.foreign`, JEP 454 in Java 22), which has no size limit.
- **Vector API** (JEP 469, incubating in Java 21+, finalized expected in 25/26) gives explicit SIMD intrinsics — `FloatVector.fromMemorySegment(...)` over a mmap'd buffer can do 8× f32 dot products per CPU instruction without leaving the JVM.
- **JNI bridges to BLAS** if you want full BLAS-accelerated matmul: OpenBLAS, Intel MKL, Apple Accelerate via `org.bytedeco.openblas` or `dev.ludovic.netlib`. ND4J / Deeplearning4j wrap these with a friendly tensor API.
- **HuggingFace tokenizers Java** binding (`ai.djl.huggingface.tokenizers`) handles the BPE.
- **Mature mmap and async I/O.** `java.nio.channels.FileChannel`, `MemorySegment`, `Arena.ofShared` etc.
- **Strong typing and refactoring.** A Java port catches a category of errors at compile time that a Python port wouldn't (matrix shape mismatches via generics or shape annotations, format dispatch via enums + sealed interfaces).

The downsides are minor:

- **No native FP4 / Q4K codec.** You'd write block dequantization code in Java (or call into a small C library via FFM API). For a toy, just use f16.
- **GPU acceleration is harder.** Java has no native Metal / CUDA; you'd go through JOCL or similar, which is significantly more work than the Rust Metal path. For a toy, CPU-only is fine.
- **Slightly larger binary.** GraalVM native-image can produce a single executable, but the JIT-warmup cost on the JVM means a one-shot CLI invocation is slower than Rust. For a long-running REPL or server it doesn't matter.

### Sketch of a Java architecture

The crate-level structure of LARQL maps cleanly to Java packages. A reasonable layout for a toy version:

```
larql-toy/                                Maven or Gradle project
├── pom.xml
├── src/main/java/io/larql/toy/
│   ├── arch/                             // Architecture detection + tensor naming
│   │   ├── ModelArchitecture.java
│   │   ├── Gpt2Architecture.java
│   │   ├── LlamaArchitecture.java
│   │   └── ArchitectureRegistry.java
│   ├── compute/                          // Matmul + activation kernels
│   │   ├── MatMul.java                   // BLAS via netlib-java OR Vector API
│   │   ├── Activations.java              // gelu, silu, gelu_tanh
│   │   ├── Norms.java                    // rms_norm, layer_norm
│   │   └── TopK.java                     // O(K)-memory min-heap top-K
│   ├── safetensors/                      // Weight file readers
│   │   ├── SafetensorsFile.java          // mmap + JSON header + tensor view
│   │   ├── TensorView.java               // float16 / float32 access
│   │   └── DTypes.java                   // f16 ↔ f32 conversion
│   ├── tokenizer/                        // BPE wrapper
│   │   └── Tokenizer.java                // wraps DJL tokenizers
│   ├── vindex/                           // The graph storage
│   │   ├── Vindex.java                   // top-level: load, save
│   │   ├── VindexConfig.java             // index.json model
│   │   ├── GateStore.java                // mmap'd gate matrix per layer
│   │   ├── DownMeta.java                 // per-feature top-K vocab metadata
│   │   ├── PatchedVindex.java            // overlay (patch on top of base)
│   │   ├── GateKnn.java                  // KNN query: gate_matrix @ x → top-K
│   │   └── Walk.java                     // multi-layer KNN trace
│   ├── inference/                        // Forward pass
│   │   ├── ModelWeights.java             // loaded weights container
│   │   ├── Attention.java                // QKV / softmax / O / multi-head
│   │   ├── DenseFfn.java                 // W_out · GELU(W_in · x) — for GPT-2
│   │   ├── GatedFfn.java                 // W_down · (σ(W_gate · x) ⊙ (W_up · x))
│   │   ├── WalkFfn.java                  // Sparse FFN via gate KNN + per-feat sum
│   │   ├── ForwardPass.java              // Embed → layers → unembed → softmax
│   │   ├── Trace.java                    // Optional residual capture per layer
│   │   └── Hooks.java                    // LayerHook trait for ablate / steer
│   ├── extract/                          // Build a vindex from safetensors
│   │   ├── Extractor.java                // Streaming extract: layer-by-layer
│   │   └── DownMetaBuilder.java          // Compute top-K vocab per feature
│   ├── lql/                              // The query language
│   │   ├── Lexer.java                    // Hand-rolled tokenizer
│   │   ├── Parser.java                   // Recursive-descent parser
│   │   ├── Ast.java                      // Sealed-interface statement tree
│   │   ├── Executor.java                 // Walks the AST against a Vindex
│   │   └── Repl.java                     // Interactive shell
│   ├── mutate/                           // INSERT / DELETE / UPDATE primitives
│   │   ├── Constellation.java            // Multi-layer insert via residual capture
│   │   └── EdgeInstall.java              // The install_edge primitive
│   ├── compile/                          // Write back to safetensors
│   │   └── CompileToModel.java
│   └── cli/                              // Command-line front-end
│       ├── Main.java
│       └── commands/
│           ├── ExtractCmd.java
│           ├── DescribeCmd.java
│           ├── InferCmd.java
│           ├── InsertCmd.java
│           └── CompileCmd.java
└── src/test/java/io/larql/toy/
    ├── inference/WalkFfnEquivalenceTest.java       // The boundary sweep
    ├── mutate/ConstellationInsertTest.java         // Atlantis → Poseidon
    └── compile/RoundTripTest.java                  // Insert → compile → load → infer
```

### The five most interesting Java implementations

#### 1. Memory-mapping the gate file

Java 22+ FFM API:

```java
public final class GateStore implements AutoCloseable {
    private final Arena arena = Arena.ofShared();
    private final MemorySegment data;          // mmap'd gate_vectors.bin
    private final long[] layerOffsets;         // float-offsets per layer
    private final int hidden;
    private final boolean f16;

    public GateStore(Path gateFile, VindexConfig cfg) throws IOException {
        try (var ch = FileChannel.open(gateFile, StandardOpenOption.READ)) {
            this.data = ch.map(MapMode.READ_ONLY, 0, ch.size(), arena);
        }
        this.hidden = cfg.hiddenSize();
        this.f16 = cfg.dtype() == DType.F16;
        this.layerOffsets = cfg.layerOffsets();
    }

    public MemorySegment layerView(int layer) {
        long byteOffset = layerOffsets[layer] * (f16 ? 2L : 4L);
        long byteLength = cfg.numFeatures(layer) * (long) hidden * (f16 ? 2L : 4L);
        return data.asSlice(byteOffset, byteLength);
    }

    public void close() { arena.close(); }
}
```

The OS demand-pages whatever you read; sleeping pages cost zero RAM. Same memory profile as Rust's `memmap2::Mmap`.

#### 2. Gate KNN with the Vector API

```java
public final class GateKnn {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static List<int[]> topK(MemorySegment gate, float[] residual,
                                   int numFeatures, int hidden, int topK) {
        float[] scores = new float[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            long rowStart = (long) i * hidden * 4L;          // f32 layout
            float acc = 0.0f;
            int j = 0;
            int upper = SPECIES.loopBound(hidden);
            FloatVector vSum = FloatVector.zero(SPECIES);
            for (; j < upper; j += SPECIES.length()) {
                FloatVector vG = FloatVector.fromMemorySegment(SPECIES, gate,
                    rowStart + j * 4L, ByteOrder.nativeOrder());
                FloatVector vR = FloatVector.fromArray(SPECIES, residual, j);
                vSum = vG.fma(vR, vSum);
            }
            acc = vSum.reduceLanes(VectorOperators.ADD);
            for (; j < hidden; j++) {
                acc += gate.get(ValueLayout.JAVA_FLOAT, rowStart + j * 4L) * residual[j];
            }
            scores[i] = acc;
        }
        return TopK.byAbs(scores, topK);
    }
}
```

For a real implementation the per-feature loop becomes a single BLAS gemv via `dev.ludovic.netlib.Blas` — much faster, but the Vector API path works without native dependencies and is plenty fast for GPT-2.

#### 3. Sparse FFN walk

```java
public Tensor walkFfn(int layer, Tensor x /* [seq, hidden] */) {
    int seq = x.shape(0), hidden = x.shape(1);
    Tensor out = Tensor.zeros(seq, hidden);
    for (int s = 0; s < seq; s++) {
        float[] xs = x.row(s);
        var hits = vindex.gateKnn(layer, xs, topK);              // [(feat, score)]
        for (var hit : hits) {
            int i = hit.feature();
            float gateScore = hit.score();
            float upScore = weights.upRow(layer, i).dot(xs);     // u_i · x
            float a = activations.silu(gateScore) * upScore;
            // out[s] += a * down_col(layer, i)
            weights.downCol(layer, i).addScaledTo(out.row(s), a);
        }
    }
    return out;
}
```

Two methods on `ModelWeights` — `upRow(layer, i)` and `downCol(layer, i)` — return small `FloatTensor` views directly into the mmap'd weight files. No allocation in the inner loop other than `out`.

#### 4. The constellation insert

```java
public class Constellation {
    public void insertFact(Vindex vindex, String entity, String relation,
                           String target, IntStream installLayers,
                           float alpha) {
        String prompt = "The " + relation + " of " + entity + " is";
        int[] tokens = tokenizer.encode(prompt);
        var residuals = forward.capture(tokens, installLayers.toArray());

        float[] targetEmbed = vindex.embed(target);
        Vec.l2Normalize(targetEmbed);

        installLayers.forEach(L -> {
            int slot = vindex.findFreeFeature(L);
            float[] r = residuals.atLayer(L);
            float[] rUnit = Vec.l2Normalize(r.clone());

            float gNorm = vindex.medianGateNorm(L);
            float uNorm = vindex.medianUpNorm(L);
            float dNorm = vindex.medianDownNorm(L);

            float[] gateVec = Vec.scale(rUnit, gNorm * GATE_SCALE);
            float[] upVec   = Vec.scale(rUnit, uNorm);
            float[] downVec = Vec.scale(targetEmbed, dNorm * alpha);

            vindex.setGate(L, slot, gateVec);
            vindex.setUp  (L, slot, upVec);
            vindex.setDown(L, slot, downVec);

            FeatureMeta meta = new FeatureMeta(target, vindex.tokenId(target),
                                               /* score */ 0.95f);
            vindex.setMeta(L, slot, meta);
        });

        // Optional: refine via Gram-Schmidt against decoys.
        // ...
    }
}
```

This is a near-direct port of the Rust `compose.rs` shown in `03-software-architecture.md`. ~80 lines including helpers.

#### 5. The walk-FFN equivalence test

```java
@Test public void walkMatchesDenseAtEveryBoundary() {
    var weights = ModelWeights.load(GPT2_PATH);
    var vindex  = Vindex.load(GPT2_VINDEX);

    String[] prompts = {
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Japan is",
    };

    for (int boundary = 0; boundary <= weights.numLayers(); boundary++) {
        var dense = new ForwardPass(weights, vindex,
                                    /* walk_from_layer= */ Integer.MAX_VALUE);
        var hybrid = new ForwardPass(weights, vindex,
                                     /* walk_from_layer= */ boundary);
        for (var prompt : prompts) {
            int[] tokens = tokenizer.encode(prompt);
            int denseTop1 = dense.predict(tokens).top1();
            int hybridTop1 = hybrid.predict(tokens).top1();
            assertEquals(denseTop1, hybridTop1,
                "boundary L%d, prompt '%s'".formatted(boundary, prompt));
        }
    }
}
```

This single test is the rigour anchor — it proves the walk is equivalent to dense at every layer-mix. If this test passes on a toy GPT-2 implementation, the rest of the pipeline (DESCRIBE, INSERT, COMPILE) is correctness-assured at the FFN substitution level.

### Estimated effort

For a single developer experienced with both Java and transformer internals:

- **Week 1.** Safetensors reader + GPT-2 forward pass (verified against HuggingFace's outputs). Walk-FFN sparse path. Boundary sweep test passes.
- **Week 2.** Vindex extract + DESCRIBE + LQL parser/executor (4–5 statement types). REPL.
- **Week 3.** Constellation INSERT + compile back to safetensors. End-to-end test (Atlantis → Poseidon round-trip via Transformers).
- **Week 4.** Polish: error handling, CLI UX, docs, examples. Maybe HNSW for browse acceleration. Maybe Apache MIME-served `larql-toy serve`.

A month of focused work, ~3,000 lines of well-tested Java. The result would have ~80% of the headline functionality of LARQL on small dense models. Notably absent: GPU, MoE, Q4K, FP4, sharding, hooks-during-generation, the 2,000-test Rust validation suite. But the **conceptual demonstration** — model is a graph database, edits are training-free, COMPILE round-trips to standard model files — would all be there.

### Why Java in particular is interesting

A Java implementation has one strategic advantage: **distribution to Android, JVM-on-Hadoop, Kotlin/server backends, embedded JVM-on-edge devices**. The vast majority of enterprise Java backends are inference-naive today; a Java-native LLM toolchain that runs on existing infrastructure (no Python deployment, no Docker, no GPU) has a different go-to-market than a Rust / Python tool.

The Rust ecosystem (LARQL's primary target) is good for performance and embeddability into systems contexts. A Java sibling is good for *enterprise reach*. Both can share `.vindex` files and `.vlp` patches as portable artifacts. The vindex format spec (`vindex-format-spec.md`) is detailed enough that a Java reader is a mechanical translation, not a research project.

The ideal split, if both ecosystems were active:

- **Rust LARQL** — production inference, GPU acceleration, Q4K/FP4, distributed serving, mech-interp engine.
- **Java LARQL** — enterprise deployments, JVM-native pipelines, model catalogs in Hadoop/Spark, edit-and-ship workflows from the same JVM as a webapp.

The vindex artifacts (the directory of mmap'd files) are the lingua franca; both implementations consume the same bytes. This is the same shape as today's safetensors / GGUF separation — different runtimes, same files — but with the vindex format adding the graph-query and edit dimensions on top.

---

## Summary

- **The graph view is the more direct view of FFN computation.** Whether matmul gets *replaced* by traversal is an empirical question of when sparsity, bandwidth, and predictable indexing win out over BLAS — today they tie at ~80% K, and the conceptual gains (inspect / edit / distribute) outweigh the modest perf wins.
- **The MVP is small.** ~1,500–2,000 lines of Python (or ~3,000 of Java) cover the headline demo: extract a vindex, browse it as a graph, walk inference, insert a fact, COMPILE back to safetensors.
- **A Java toy on GPT-2 / nanochat is a 4-week project.** mmap, Vector API, BLAS bindings, and DJL tokenizers cover the substrate. The architecture mirrors LARQL's crate layout one-to-one. The boundary-sweep test gives correctness rigour. The result demonstrates the central thesis on a model anyone can run on a laptop.

The point of the toy isn't to compete with LARQL on production capabilities — it's to make the underlying technique reproducible, teachable, and portable. A Java port that runs on an enterprise JVM, talks to the same `.vindex` files as the Rust system, and lets a developer say *"here is my model, here is the new fact, COMPILE, ship"* would be a meaningful contribution to the open-source ecosystem.
