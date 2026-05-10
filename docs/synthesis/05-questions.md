# LARQL: Questions and Answers

> Direct answers to three questions about whether the graph view is the
> "real" picture, how to compress the technique to something tractable,
> and whether a Java toy on GPT-2 / nanochat is practical.

---

## Q1. Is the graph-network view a more direct understanding of what LLMs actually do, and could matrix multiplies be replaced by efficient graph traversal?

### Short answer

**Partially yes, with important caveats.** The graph view *is* a more
faithful description of what the FFN does on a per-token basis.
Replacing matrix multiplies with graph traversal in the literal sense
(pointer-chasing through edges) would be slower than current
implementations, not faster. The realistic interpretation is:

> The FFN is a top-K sparse dispatch into a static graph of features.
> The cost-saving opportunity is in **avoiding the large dense
> down-matmul**, not in eliminating BLAS altogether.

LARQL's "walk" already demonstrates this. It is fractionally faster
than dense on a single CPU and dramatically smaller in memory traffic
once `K ≪ I`. But it is *still* implemented as BLAS calls — just
sparser ones — and it does *not* attempt to replace attention.

### Longer answer

#### Where the graph view is genuinely more direct

The dense FFN equation

```
y = W_down ( σ(W_gate · x) ⊙ (W_up · x) )
```

obscures something the graph reading makes obvious: **only a small
number of features per token actually contribute**. Empirically on
Gemma 3-4B, K ≈ 10–100 of the 10,240 features per layer is enough to
recover the full dense output to within float-noise. The remaining
≈99% of the matrix multiply is computing zeros (or near-zeros, killed
by SiLU) and then summing them.

This is true at the *math* level — equation (2) in
[`02-mathematical-foundation.md`](02-mathematical-foundation.md) — and
the heavy-tailed activation distribution is a property of trained
transformers, not an artefact of any particular implementation.

So in this sense yes: the FFN is "really" a graph dispatch, and the
matrix-multiply formulation is computing far more than it needs to.

#### Where the matmul formulation is still right

Three reasons it isn't a straightforward win to "replace matmul with
graph traversal":

1. **Top-K KNN itself is a matmul.** To find the top-K gate features
   for an input `x`, you have to compare `x` to all `I` gate vectors.
   The straightforward way is `W_gate · x` — the *exact* matmul we
   were trying to replace. LARQL has approximate KNN via HNSW (the
   `gate_mmap_bytes` plus a hierarchical index), but in production
   the hot path still uses fused BLAS gemv because `K = 10–100` and
   `I = 10,240` is small enough that the dense gemv beats indexed
   lookup on cache locality.

2. **Pointer-chasing is bad on modern hardware.** Real graph traversal
   (follow edges, dereference, repeat) is exactly what GPUs and modern
   CPUs are not built for: random-access, branchy, hostile to
   prefetching and SIMD. The win in LARQL's walk comes from
   *cache-friendly batch reads* of selected columns from a feature-major
   mmap'd file — that's still BLAS, just on a sub-matrix.

3. **Attention is not a graph dispatch in the same sense.** The FFN
   features are static (they live in the weights). Attention edges are
   *dynamically computed per token*, and they still benefit from dense
   QK^⊤ and softmax. LARQL doesn't currently graph-ify attention; the
   research roadmap (experiments 16, 18, 24, 38) explores low-rank
   decompositions of `W_Q W_K^⊤` and OV circuits, but that is much
   harder than the FFN case.

#### Where the cost saving actually appears

The walk saves cost in three concrete places:

- **Down projection becomes sparse.** Instead of `H × I` dense gemm
  (Gemma 3-4B: 2560 × 10240 = 26 M multiplies/layer), the walk does
  `H × K` (e.g. 2560 × 50 = 128 K multiplies/layer). That's ≈200×
  fewer multiplies in the down step.
- **Memory bandwidth for down weights drops.** Reading `K` columns
  instead of all `I` columns takes ~K/I of the bandwidth.
- **Many features can be skipped entirely once gate ≤ 0.** SiLU
  drives those to zero; storing them only as gate vectors (and
  loading down columns on-demand from mmap) means you don't pay for
  them until they fire.

That's a real win. But it's measured against dense CPU FFN, not
against a GPU running tile-fused matmul. The Metal GPU path on M3 Max
runs Gemma 3-4B at ~83 tok/s; the CPU walk runs at ~2 tok/s. **The
walk's advantage is qualitative — queryability, editability, no-GPU
deployment — not raw throughput.**

#### Could it be made dramatically faster?

Three plausible directions:

1. **Sub-linear gate KNN.** A learned product-quantisation index over
   gate vectors, or a hierarchy that prunes layers (HNSW, which LARQL
   already supports as a build-time option), could bring the gate
   step from `O(I H)` to `O(K log I)`. Expected speed-up: maybe 10–20×
   on the gate step at modest top-K accuracy cost.
2. **Manifold-compressed gates.** Experiment 02 in the repo found 99%
   variance in ~15 dimensions across knowledge-band gate vectors.
   Storing gates in a learned 15-D basis would reduce the gate file
   ~150× and make the KNN scan correspondingly cheaper.
3. **Attention as a graph too.** If the OV circuits can be
   factorised into static head-templates plus a small dynamic
   correction, attention also becomes a sparse dispatch, and the
   whole forward pass collapses to a sequence of small graph lookups
   plus a few low-rank corrections. Speculative; that is what the
   research experiments are aiming at.

Combine all three and you might plausibly run 4 B-parameter models on
a CPU at 50–100 tok/s, which would be transformative for deployment
economics. None of those have shipped yet.

### Verdict on the "current take"

You said:

> *"…the vast number of matrix multiplies currently used to implement
> the functionality may be highly inefficient and could potentially be
> replaced with a much more efficient approach of direct graph
> traversal."*

The first half is correct: most of the FFN matmul is computing zeros
on a per-token basis, and the heavy-tailed sparsity is a real
phenomenon. The second half is half-right: the *replacement* won't
look like literal pointer-chasing; it will look like (a) sub-linear
top-K selection plus (b) sparse cache-friendly gather-and-multiply.
The current dense-matmul formulation is not "wrong" but it is
performing a great deal of redundant work that the graph view makes
visible and exploitable. That redundancy is real value left on the
table.

---

## Q2. Can we boil it down to something simple that could be implemented in a relatively compact amount of code?

**Yes.** Stripped to its essentials, the technique is about 500–1,000
lines of code in any language plus a tokenizer and standard attention.
Here is the entire algorithmic surface needed to build a working
implementation.

### Minimal data the implementation needs

For each transformer layer ℓ, three matrices and one bias:

```
W_gate[ℓ] : (I, H)    # gate rows = triggers
W_up[ℓ]   : (I, H)    # up rows = gains
W_down[ℓ] : (H, I)    # down columns = writes
norms[ℓ]  : RMSNorm or LayerNorm parameters
```

Plus the embedding/un-embedding matrix `E : (V, H)` and standard
attention weights `W_Q, W_K, W_V, W_O` per layer. Tied LM head
(`LMHead = Eᵀ`) is the easy case.

### Six functions you need

Including only the LARQL-specific pieces (attention is just
"do attention as usual"):

```python
def gate_knn(W_gate_layer, x, K):
    """Top-K features at this layer for residual x."""
    scores = W_gate_layer @ x                # (I,)  -- gemv
    idx    = argpartition(|scores|, K)       # top-K by absolute value
    return idx, scores[idx]                  # selected feature ids + scores

def walk_ffn(W_gate, W_up, W_down, x, K, sigma=silu):
    """One FFN layer evaluated via top-K walk."""
    idx, gate_scores = gate_knn(W_gate, x, K)
    up_scores  = W_up[idx] @ x               # (K,)
    activation = sigma(gate_scores) * up_scores  # (K,)
    return W_down[:, idx] @ activation       # (H,)  -- sparse gather + gemv

def install_edge(W_gate, W_up, W_down, slot, trigger, write,
                 gate_scale=30.0, alpha_mul=1.0):
    """Write one (gate, up, down) triple at slot `slot`."""
    g_norm = ||W_gate[slot]||;  u_norm = ||W_up[slot]||;  d_norm = ||W_down[:, slot]||
    W_gate[slot]    = trigger / ||trigger|| * g_norm * gate_scale
    W_up[slot]      = trigger / ||trigger|| * u_norm
    W_down[:, slot] = write   / ||write||   * d_norm * alpha_mul

def label_feature(W_gate, W_down, layer, feat, E, K=10):
    """Project gate row and down column into embedding space."""
    g, d = W_gate[layer, feat], W_down[layer, :, feat]
    in_topk  = topk_dot(E, g, K)             # tokens whose embeddings best match the gate
    out_topk = topk_dot(E, d, K)             # tokens whose embeddings best match the down
    return in_topk, out_topk

def trace_residual(model, prompt, layers):
    """Capture the residual at chosen layers."""
    x = embed(tokens(prompt))
    captured = {}
    for ℓ in range(L):
        x = x + attention(model[ℓ], x)
        x = x + walk_ffn(... at layer ℓ ...)
        if ℓ in layers: captured[ℓ] = x.copy()
    return captured

def insert_constellation(model, trigger_prompt, target_token,
                         layers=range(20, 28), alpha=0.25):
    """Training-free knowledge insertion."""
    residuals = trace_residual(model, trigger_prompt, layers)
    embed_scale = sqrt(model.H)
    write_dir   = E[target_token] * embed_scale * alpha
    for ℓ in layers:
        slot      = find_free_feature(model[ℓ])      # any unused / low-c slot
        trigger_ℓ = residuals[ℓ]                     # already in residual space
        install_edge(model.W_gate[ℓ], model.W_up[ℓ], model.W_down[ℓ],
                     slot, trigger_ℓ, write_dir,
                     gate_scale=30.0, alpha_mul=1.0)
```

That's the entire LARQL feature set in pseudocode — about 60 lines
including comments. The non-trivial code in the real implementation
is the *infrastructure*: mmap'd file formats, patch overlays, BLAS
dispatch, GPU kernels, the LQL parser, the HuggingFace publish flow,
the streaming extractor. The *technique* itself is small.

### A genuinely minimal implementation

To prove out the technique against a real model you need:

| Component | Lines | Notes |
|-----------|------:|-------|
| Tokeniser (BPE) | 200 | Or wrap an existing library |
| Safetensors loader | 100 | Just enough to read tensors by name |
| Standard attention (multi-head, KV cache) | 150 | Boilerplate, well-known |
| RMSNorm / LayerNorm | 30 | Trivial |
| `walk_ffn`, `gate_knn` | 80 | Including tied-fast paths |
| `label_feature` | 40 | Embedding-space KNN |
| `install_edge`, `insert_constellation` | 80 | The math is in pseudocode above |
| `trace`, capture hooks | 100 | Record residuals at chosen layers |
| Top-level `forward` | 80 | Orchestrator |
| Driver / REPL | 100 | Read prompts, print top-K |
| **Total** | **~960** | Single language, single file practical |

That's a from-scratch pedagogical implementation. The full LARQL
ships ~50 KLOC of Rust, but most of that is production engineering
(GPU paths, distributed sharding, server, gRPC, multiple model
families, eight quantisation formats, …) layered on top of this
kernel.

### What you need to *not* implement

These are LARQL features you can omit for a toy:

- All the quantisation formats (Q4_K, Q6_K, MXFP4, FP4/FP8 blocks).
  Just use f32.
- The patch overlay / `.vlp` JSON format. Mutate the in-memory
  tensors directly and re-save when done.
- Distributed sharding, MoE routing, expert servers.
- The HNSW index. Linear gate-KNN scan is fine for a toy.
- The full LQL parser. A handful of methods on a model object
  (`describe`, `walk`, `infer`, `insert`, `compile`) is enough.

---

## Q3. How practical is a Java toy on GPT-2 or nanochat? Sketch the architecture.

### Practicality

**Practical and well-suited.** Both target models fit on a laptop, the
math is easy, and Java's array-and-buffer story is excellent for this
kind of work. The two architectural details to mind are:

1. **GPT-2 has an ungated FFN.** It's `down(GELU(up(x)))`, with no
   `W_gate` and no element-wise product. The graph reading still
   works — each feature's *trigger* becomes the up row `u_i` and its
   activation is just `GELU(u_i · x)` — but the LARQL primitives
   need a small fork for the ungated case. Top-K selection by
   `|GELU(u_i · x)|` is the natural analogue of gate-KNN.

2. **nanochat depends on the version.** Karpathy's recent work
   typically uses a Llama-style architecture (RMSNorm, RoPE, SwiGLU
   gated FFN), in which case the LARQL primitives apply directly.
   Older nanoGPT / minGPT used GPT-2-style ungated FFN. Check the
   specific release, but expect the modern variant to be gated.

GPT-2-small is 117 M parameters, 12 layers, hidden=768, intermediate=3072.
A nanochat-sized model is in the same order. Both:

- Fit comfortably in a few GB of RAM.
- Run inference in real-time on a single CPU thread.
- Have well-understood weight layouts and tokenisers.

That makes them ideal targets for a from-scratch toy.

### Java fitness

Java is a fine choice for the toy. Key strengths:

- **Memory-mapped files.** `FileChannel.map()` plus
  `MappedByteBuffer` give zero-copy access to safetensors on disk.
  This is what makes LARQL's mmap-first design feasible in Rust;
  the same pattern works in Java with `LongBuffer.asFloatBuffer()`.
- **The Vector API (jdk.incubator.vector).** Modern Java has
  SIMD primitives for `FloatVector` operations, sufficient for
  hand-written gemv at competitive performance. For a toy, plain
  `float[]` arithmetic is fine.
- **Existing libraries** if you don't want to roll your own:
  - DJL (Deep Java Library) — has tokenizers, ONNX runtime,
    safetensors readers.
  - JBlas / ND4J — BLAS-backed matrix libraries.
  - HuggingFace `tokenizers` Java bindings — for BPE.
- **Easy interop with Python tooling** for cross-validation:
  load the same model in transformers, dump intermediate
  activations, compare against your Java forward pass.

The downside is that Java will be ~3–5× slower than equivalent C
or Rust on raw float math, but for a toy on a 117 M model this is
irrelevant — you're trading 50 ms/token for 200 ms/token, both are
"interactive."

### Sketch architecture for a Java toy

Module layout:

```
toy-larql-java/
├── pom.xml
├── src/main/java/io/larql/toy/
│   ├── Main.java                    REPL entry point
│   ├── tokenizer/
│   │   └── BpeTokenizer.java        Wraps DJL or HF tokenizers Java
│   ├── model/
│   │   ├── ModelConfig.java         hidden, intermediate, layers, vocab
│   │   ├── SafetensorsReader.java   FileChannel.map + offset table parser
│   │   └── Tensor2D.java            wraps a FloatBuffer, knows shape, gemv
│   ├── vindex/
│   │   ├── Vindex.java              the in-memory model graph
│   │   ├── Layer.java               W_gate, W_up, W_down, norms per layer
│   │   ├── FeatureMeta.java         per-feature label/confidence/selectivity
│   │   └── Patch.java               JSON-serialisable list of edits
│   ├── compute/
│   │   ├── Attention.java           BLAS-fused multi-head + KV cache
│   │   ├── Norm.java                RMSNorm and LayerNorm
│   │   ├── Activation.java          SiLU, GELU
│   │   └── WalkFfn.java             gate_knn + sparse down (the heart)
│   ├── interpret/
│   │   ├── Labeler.java             embedding-space top-K for any vector
│   │   ├── Tracer.java              capture residuals at chosen layers
│   │   └── ConstellationInserter.java  trigger capture + install_edge ×N
│   ├── query/
│   │   ├── Lql.java                 mini-lexer/parser for a subset of LQL
│   │   └── Executor.java            dispatch DESCRIBE/WALK/INFER/INSERT
│   └── util/
│       ├── TopK.java                bounded heap for top-K selection
│       └── Vectors.java             dot, axpy, l2norm, normalise
└── src/test/...                     JUnit tests + a sample GPT-2-small fixture
```

Public-facing API (illustrative — adapt to taste):

```java
// Loading
Vindex v = Vindex.load(Path.of("gpt2-small.vindex"));   // mmap'd directory
// (Or load straight from safetensors for a quick toy)
Vindex v = Vindex.fromSafetensors(Path.of("gpt2-small.safetensors"));

// Browse
List<Edge> edges = v.describe("France");
List<TopKHit> firing = v.walk("The capital of France is", /*topK=*/10);

// Inference
List<TokenProb> top = v.infer("The capital of France is", /*topK=*/3);

// Trace
Trace t = v.trace("The capital of France is");
t.forTarget("Paris").print();   // per-layer rank, prob, attn delta, ffn delta

// Edit
v.insertEdge("Atlantis", "capital-of", "Poseidon");   // constellation default
v.savePatch(Path.of("atlantis.vlp"));

// Compile
v.compileToSafetensors(Path.of("gpt2-edited.safetensors"));
```

### Specific implementation notes

- **GPT-2-small dimensions** make brute-force top-K trivial:
  `intermediate = 3072`, so a full gate-KNN scan is 3072 dot products
  per layer per token. Even at 768 hidden, that's < 0.5 ms per layer
  in plain Java.
- **Attention already exists** as multi-head causal self-attention
  with a KV cache. There is nothing LARQL-specific about implementing
  this; copy from any "GPT-2 in pure X" reference.
- **For ungated FFN (GPT-2 proper),** treat the up matrix as both
  trigger and gain. The sparse formula simplifies to:
  ```
  scores  = W_up · x                # (I,) — gemv
  picks   = top_K(|GELU(scores)|)
  acts    = GELU(scores[picks])     # (K,)
  result  = W_down[:, picks] · acts # (H,) — sparse gemm
  ```
  This is even simpler than the gated case. For nanochat (gated),
  use the full gated form from `WalkFfn` in the math doc.
- **Tokenizer** is the largest non-LARQL dependency. The cleanest
  path is `ai.djl:djl-api` plus
  `ai.djl.huggingface:tokenizers`, which can directly load
  `tokenizer.json` for both GPT-2 and nanochat.
- **`install_edge` in Java** is a few `for` loops. Magnitude
  preservation is direct from `02-mathematical-foundation.md`
  equation (19).
- **Safetensors loader.** It's a JSON header followed by raw float
  blobs. About 100 lines of Java.

### What to validate first

Build the toy in this order so each step is independently provable:

1. **Tokenize → embed → forward (dense FFN) → de-tokenize.**
   Reproduce `transformers.GPT2LMHeadModel.generate()` output bit-for-bit
   on a few short prompts. This proves you've got the architecture
   right before any LARQL-specific changes.

2. **Replace dense FFN with `walk_ffn` at K = I (i.e. all features).**
   Output should be identical to step 1. This proves your sparse
   path is correct in the easy regime.

3. **Reduce K and measure divergence.** At K = 100, K = 50, K = 10,
   measure top-1 token agreement with step 1 across a held-out set
   of prompts. You should see ≥99% agreement at K = 50 on GPT-2.

4. **Implement `label_feature` and `describe`.** For a small
   sample of features at a single layer, verify the top-K embedding
   matches expected outputs (e.g. the most-fired feature on
   "the capital of France" should label cleanly).

5. **Implement `install_edge` and a single-fact insert.** Try
   inserting `(Atlantis, capital-of, Poseidon)` with the
   constellation default, verify the top-1 prediction for
   "the capital of Atlantis is" shifts toward Poseidon, and
   verify France→Paris is preserved (with measurable degradation).

6. **Compile back to safetensors.** Reload the edited model in
   `transformers` and verify the same predictions outside Java.

If steps 1–6 succeed end-to-end on GPT-2-small, you have a complete,
self-contained reproduction of the LARQL technique in Java in
roughly the line count estimated in Q2. nanochat-sized models
add a tokenizer/architecture variant but no new mathematics.

### Estimated effort

Rough estimate, with familiarity with Java and transformer
internals:

- Steps 1–2 (forward pass, walk-equivalent): **2–3 days.**
- Steps 3–4 (top-K reduction, labels): **1 day.**
- Step 5 (constellation insert): **2 days,** the trickiest part
  because you need clean residual capture and norm preservation.
- Step 6 (compile back): **1 day.**

Call it a focused two-week project for a working toy, plus another
week to polish into something demoable. The Rust LARQL got to its
current scale via a very large amount of additional engineering
(quantisation, GPU, distributed, multi-architecture, mechanistic-interp
hooks), but the *core technique* is reproducible at the scale
sketched here.

### One concrete pitfall to flag

The constellation insert only works cleanly when:

- The model has a **tied LM head** (or you've re-derived
  `embed_scale` for an untied head).
- The captured residual at the chosen layers is **post-norm**, not
  pre-norm — gate KNN sees the post-norm residual, so that's the
  query you need to match.
- The `embed_scale = √H` constant is right for the tied-LM-head
  family. For models with different normalisation conventions you
  may need to multiply by an additional factor read from the
  RMSNorm `scale` parameters.

The Rust code handles these per-architecture; in a Java toy you
will hit them and need to recalibrate. Plan for an empirical
calibration step (sweep `α` and `gate_scale` on a held-out set
of inserts, pick the Pareto frontier — see the table in
`docs/training-free-insert.md` for the shape of the curve) rather
than trusting the Gemma constants out of the box.

---

## TL;DR

| Question | Answer |
|----------|--------|
| Is the graph view "the real" computation? | Yes for FFN — empirically, a top-K dispatch of ~10–100 features per layer reproduces dense output. No simple "graph traversal" replaces matmul, but the practical replacement (sparse cache-friendly gather + gemv, plus sub-linear KNN) leaves a real efficiency win on the table — perhaps an order of magnitude on CPU once HNSW gates and manifold compression are in. |
| Can it be made compact? | Yes — ~1,000 lines plus tokenizer / standard attention. The math is six small functions; everything else in LARQL is production engineering. |
| Java toy on GPT-2 / nanochat? | Practical, ~2 weeks for a working version. Ungated GPT-2 needs a small adaptation (trigger = up row); gated nanochat fits the LARQL primitives directly. |
