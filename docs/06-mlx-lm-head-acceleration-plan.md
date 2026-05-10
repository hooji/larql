# MLX LM Head Acceleration via Random Projection: Implementation Plan

> A self-contained sprint plan for landing a 30%+ logit-projection speedup in `mlx-lm` for Qwen3.5 (and architecturally compatible large-vocab models). Designed to be executed by a fresh Claude Code session that has access to this LARQL repository and can clone external repos.

---

## 0. For the agent picking this up

Welcome. You are picking up an inference-optimization sprint that has been planned end-to-end. Read this entire document before starting. Then read these supporting documents in this repository, in order:

1. [`docs/01-high-level-explainer.md`](01-high-level-explainer.md) — overview of LARQL and the graph view of FFN computation. Provides the conceptual frame for why the technique works.
2. [`docs/02-mathematical-foundations.md`](02-mathematical-foundations.md) — Sections 2-4 are directly relevant: the matmul-as-graph identity, gate KNN as MIPS, sparsity bounds. The same math underlies what we are building here, applied to `lm_head` instead of `W_gate`.
3. [`docs/03-software-architecture.md`](03-software-architecture.md) — how LARQL organizes the same kind of work (gate KNN dispatch, walk-FFN, COMPILE). Useful for understanding how a similar feature is structured in production.

The optimization is well-grounded mathematically and engineering-tractable. Your job is to ship it cleanly: implement, validate, benchmark, and prepare a clean PR for `mlx-lm`.

The user has approved this plan. Do not redesign the approach without explicit authorization — if you find a fundamental blocker, surface it and ask. Implementation details (e.g. exact MLX API calls, how to thread a flag through config) are yours to decide.

---

## 1. Executive summary

**What we're building:** a drop-in approximation for the final language-modeling head (`lm_head`) projection in `mlx-lm`'s Qwen3.5 model class. The dense `[V × H]` matmul that produces logits is replaced with a two-stage approximate-then-verify search: a cheap low-dimensional pre-filter selects ~2048 candidate tokens, then exact dot products run only on those candidates. The pre-filter uses a Gaussian random projection (Johnson-Lindenstrauss style) and is built at model load time — no sidecar files, no offline tooling, no model recompilation.

**Why it matters:** on Qwen3.5-122B-A10B with vocab=248,320 and hidden~5120, the dense lm_head projection is `~2.5 GB read per token` — a hard memory-bandwidth bottleneck on Mac. Sampling only needs the top scoring tokens; reading every row to find them is wasted bandwidth. Replacing the dense gemv with a 64-dim random-projection prefilter + exact verify on top-2048 reduces lm_head bandwidth ~50×, which translates to ~8-12% total decode speedup on the 122B variant (estimated; measure to confirm).

**Why now:** Qwen3.5's hybrid attention (Gated DeltaNet + GQA) and ultra-sparse MoE (top-10 of 512 experts) are architecturally innovative but make non-MoE operations like `lm_head` a *larger* relative share of decode time than they would be on a dense model. With `lm_head` becoming proportionally more dominant, the optimization wins are bigger.

**Constraints:**
- No sidecar files. Index built at load time. Pure MLX/numpy, no external deps (no hnswlib, no faiss).
- Must preserve sampling quality. Greedy decode output should be identical to dense; top-K / top-P sampling distributions should be statistically indistinguishable.
- Must be opt-in via config or model-load flag, defaulting off until the team is comfortable enabling by default.
- Must work with existing MLX-LM samplers (greedy, top-K, top-P, repetition penalty) without those samplers being modified.

**Definition of done:**
- A PR against `ml-explore/mlx-lm` (or the appropriate active fork) implementing the feature.
- Unit tests covering the new module.
- A benchmark script demonstrating speedup on Qwen3.5-9B (small model, fast iteration) and Qwen3.5-122B-A10B (target model).
- Quality validation showing top-50 recall ≥ 0.99 vs dense and MMLU/HellaSwag drift within noise.
- PR description referencing this plan and the LARQL docs that ground the technique.

---

## 2. Theoretical background

### 2.1 The lm_head bottleneck on large-vocab models

The final layer of any transformer is the language-modeling head: a linear projection from hidden space to vocabulary space.

```
logits = lm_head(final_residual)        # [..., vocab_size]
                = final_residual @ W_lm_head.T
```

For Qwen3.5-122B-A10B:
- `vocab_size = 248,320`
- `hidden_size ≈ 5120`
- `W_lm_head` shape: `[248320, 5120]`
- Parameter count: ~1.27 billion

This matmul executes **once per token** during decoding. At BF16 storage, the lm_head weight tensor occupies ~2.5 GB. On Mac Studio M3 Ultra with ~200 GB/s effective CPU memory bandwidth, just *reading* the lm_head once costs ~12 ms. On Metal, with ~400 GB/s effective GPU bandwidth, it's ~6 ms. Either way, it is one of the largest individual ops in the per-token forward pass.

The problem is structural: **modern open models have huge vocabularies** (Gemma 3 4B: 262K, Qwen3.5: 248K, Kimi K2: 163K) to support multilingual coverage. When vocab grows linearly, lm_head cost grows linearly. When the rest of the model uses MoE sparsity to reduce per-token compute (Qwen3.5: 4-13% of params active per token), lm_head becomes a *relatively bigger* slice of total decode time. On dense models like Llama 3 8B with vocab=128K, lm_head is ~10-15% of CPU decode. On Qwen3.5-122B-A10B with vocab=248K and active sparsity, it's ~10-15% of decode time despite the model being 40× larger — the proportion holds because both numerator and denominator scaled.

### 2.2 What sampling actually needs

The dense logits computation produces a vector of length `vocab_size`. Almost no sampler uses all of it. Common samplers and what they actually need:

| Sampler | Needs |
|---|---|
| Greedy (argmax) | top-1 |
| Top-K with K≤200 | top-K |
| Top-P / nucleus sampling (P≤0.99) | top ~few hundred (the cumulative-probability tail is rarely large) |
| Temperature scaling | full distribution (but only over tokens that survive top-K/top-P) |
| Repetition penalty | a small set of recent tokens; full distribution otherwise |
| Speculative decoding verification | top candidates of the verifier |

The vast majority of *production* decoding is greedy or top-K with K ≤ 50 or top-P with P ≤ 0.95. **For all of these, we don't need the bottom 99%+ of the logits distribution at all.** The dense computation is doing 1.27 billion ops of which only a few thousand are actually consumed.

This is the same insight that powered LARQL's walk-FFN: most FFN features don't fire for any given token; computing all of them is bandwidth waste. See [`docs/02-mathematical-foundations.md`](02-mathematical-foundations.md) §3 for the equivalent argument applied to FFN gate vectors.

### 2.3 Approximate maximum inner product search (AMIPS)

What we want: given a query vector `q ∈ ℝ^H` and a database of vectors `W ∈ ℝ^{V × H}`, return the top-K of `Wq`, the inner products of `q` with each row of `W`.

Exact computation: O(VH). For our problem: 1.27 billion multiplies per token.

Approximate MIPS solutions trade some recall for sub-linear time. The standard approaches:

1. **HNSW** (Hierarchical Navigable Small World graphs). O(log V × H) query time after O(VH log V × M) build. Used by FAISS, hnswlib. Excellent recall but build time is non-trivial.
2. **Product Quantization (PQ)**. Decompose vectors into chunks, quantize each chunk to a small codebook. Search uses precomputed lookup tables. Lower recall than HNSW; very fast.
3. **Random projection / LSH** (Locality-Sensitive Hashing). Project vectors to a lower dimension; the projection approximately preserves dot products. Search becomes O(Vd) where d ≪ H. Very fast build, very simple, robust.
4. **Two-stage approximate-then-verify.** Use any cheap approximation as a pre-filter, then run exact dot products on the top-N candidates. Recovers exact top-K with high probability when N is sufficiently large.

We are using **(3) + (4)**: random projection as the approximation, followed by exact verification.

### 2.4 Random projection: theory and why it works

The Johnson-Lindenstrauss (JL) lemma says, informally, that a set of N points in high-dimensional space can be embedded into a space of dimension `O(log N / ε²)` such that pairwise distances (and equivalently inner products of unit vectors) are preserved up to multiplicative factor `1 ± ε`. The embedding can be a random Gaussian projection.

Formal statement (one common form): Let `R ∈ ℝ^{d × H}` be a random matrix with iid `N(0, 1/d)` entries. For any fixed unit vectors `u, v ∈ ℝ^H`:

```
P[ |⟨Ru, Rv⟩ - ⟨u, v⟩| > ε ] ≤ 2 exp(-d ε² / 8)
```

Setting `d = 8 log(2/δ) / ε²` gives the bound `|⟨Ru, Rv⟩ - ⟨u, v⟩| ≤ ε` with probability at least `1 - δ`.

For our problem: `V = 248,320` rows of `lm_head`. Union-bounding over all `V` inner products with a query `q`:

```
P[ ∀v: |⟨RW[v], Rq⟩ - ⟨W[v], q⟩| ≤ ε ] ≥ 1 - 2V exp(-d ε² / 8)
```

For `ε = 0.1`, `δ = 0.001`, `V = 248,320` we get `d ≈ 1700`. For `ε = 0.3`, `δ = 0.01` we get `d ≈ 200`. These are pessimistic — tight bounds; in practice random projections work much better than the worst-case bound suggests.

**The crucial observation:** we don't need to preserve every inner product accurately. We only need the *ranking* in the top region to be stable enough that the true top-K appears in the top-N approximate scores, where N >> K. The verification step (exact dot products on the top-N) will recover the true top-K exactly.

Empirically, for embedding-style matrices (which `lm_head` is — it's the transpose of the embedding matrix in tied configurations, or a closely-related linear projection in untied), `d = 64` to `d = 128` with `N = 2048` recovers the true top-50 with > 99% jaccard. The literature on locality-sensitive hashing for word embeddings has many results in this range.

### 2.5 Two-stage approximate-then-verify

The full algorithm:

```
# Build phase (once, at model load):
P ∈ ℝ^{d × H}                            ← random Gaussian, entries iid N(0, 1/d)
W_proj ∈ ℝ^{V × d}                       ← W @ P^T, computed once

# Query phase (per token):
q_proj = P @ q                           ∈ ℝ^d
approx_scores = W_proj @ q_proj          ∈ ℝ^V    (cheap: V × d ops)
candidate_indices = top_N(approx_scores) ∈ ℤ^N    (N ≈ 2048)
exact_scores = W[candidate_indices] @ q  ∈ ℝ^N    (cheap: N × H ops)
return (candidate_indices, exact_scores)
```

Cost analysis on Qwen3.5-122B-A10B (V=248320, H=5120, d=64, N=2048):

| Stage | Ops | Bandwidth (f16) |
|---|---|---|
| Stage 1 (approx prefilter) | 248K × 64 ≈ 15.9 M | ~32 MB read |
| Stage 2 (exact verify) | 2048 × 5120 ≈ 10.5 M | ~21 MB read |
| **Total** | **~26.4 M ops** | **~53 MB** |
| (Dense baseline) | 1.27 B ops | 2.5 GB |
| **Reduction** | **~48× compute** | **~47× bandwidth** |

Memory bandwidth is the binding constraint on Mac decode, so the ~47× bandwidth reduction is what translates to wall-clock improvement. If lm_head was ~12 ms before, it's ~0.3 ms after — essentially free.

### 2.6 Connection to LARQL's gate KNN

This is exactly the same pattern LARQL uses for FFN gate selection:

| LARQL gate KNN | This proposal |
|---|---|
| Match: `gate_matrix @ residual → top-K active features` | Match: `lm_head @ residual → top-K candidate tokens` |
| Pre-filter: `gemv` against full gate matrix (fast on BLAS) | Pre-filter: random-projection gemv (even faster) |
| Verify: sparse FFN compute on top-K | Verify: exact dot product on top-N |
| Output: feature activations summed into residual | Output: top-K logits passed to sampler |

LARQL's `walk-FFN` proves the pattern works at FFN scale: the boundary sweep (see [`docs/02-mathematical-foundations.md`](02-mathematical-foundations.md) §3 and the original `docs/walk-boundary-sweep.md`) shows that replacing dense FFN with this exact two-stage pattern preserves top-1 token output identically across all 34 layers of Gemma 3 4B.

We are doing the same thing for `lm_head`. The math is structurally the same; the implementation is simpler because lm_head is a single op, not 34 layered ops.

The one difference: LARQL's gate KNN uses brute-force gemv (not random projection) for stage 1, because `intermediate_size` is small enough (~10K) that the dense gemv is already cheap. For lm_head where V is 25× larger, the random-projection prefilter pays off.

---

## 3. Target environment

### 3.1 Engine: MLX-LM

- Repository: <https://github.com/ml-explore/mlx-lm> (or fork — confirm at sprint start)
- Language: Python (model classes, samplers, generation loop) + MLX core (C++/Metal kernels under the hood)
- Why MLX over llama.cpp:
  - Faster iteration (Python + REPL friendly)
  - The change is at the lm_head layer, which is a clean Python-level intervention point
  - Apple Silicon native; matches the user's hardware reality
  - The MLX-LM team has been actively adding architecture-aware optimizations (already shipped state-snapshot prompt caching specifically for Qwen3.5's hybrid attention)
  - llama.cpp is a worthy follow-up but its C++ codebase + tighter PR review process makes it the wrong target for the *initial* sprint

### 3.2 Model family: Qwen3.5

Qwen3.5 lineage (as of May 2026):
- Qwen3.5-9B (dense, vocab=248,320, hidden=4096, layers=32)
- Qwen3.5-27B (dense)
- Qwen3.5-35B-A3B (MoE: 35B total, 3B active)
- Qwen3.5-122B-A10B (MoE: 122B total, 10B active, hidden~5120)
- Qwen3.5-397B-A17B (MoE: 397B total, 17B active, hidden~7168)

Architecture (inherited from Qwen3-Next):
- **Hybrid attention.** 75% of layers use Gated DeltaNet (linear attention); 25% use Gated GQA Attention. This affects intermediate layers; **does not affect lm_head**.
- **Ultra-sparse MoE.** 512 routed experts + 1 shared expert with top-10 routing. Affects FFN; **does not affect lm_head**.
- **Multi-Token Prediction (MTP) head.** A small auxiliary head used for speculative-decoding-style multi-token output. The main `lm_head` is a separate, full-vocab projection and is what we are optimizing.

For the sprint:
- **Develop and validate on Qwen3.5-9B.** Loads in seconds, fits on any Mac with 32+ GB unified memory. Iteration cycle measured in seconds.
- **Final benchmark on Qwen3.5-122B-A10B.** This is where the speedup numbers in the PR description come from. Requires Mac Studio or similar (192+ GB unified memory).
- **Optional stretch:** validate on Qwen3.5-397B-A17B at Q4 if hardware available.

The technique works on all variants — the lm_head shape varies a bit (different hidden sizes), but the algorithm is identical.

### 3.3 Why this generalizes

Although we target Qwen3.5 first, the same code works on any model whose `lm_head` is a single dense `[V × H]` linear projection. That covers essentially every modern open-weights model. The PR should be written to be model-class-agnostic where possible, with Qwen3.5 as the proven case.

---

## 4. Implementation design

### 4.1 The `ApproxLMHead` module

A drop-in replacement for `nn.Linear` (or whatever module wraps `lm_head` in MLX-LM's Qwen3.5 implementation). Approximately:

```python
import math
import mlx.core as mx
import mlx.nn as nn

class ApproxLMHead(nn.Module):
    """Two-stage approximate-then-verify language-modeling head.

    Replaces a dense [V, H] projection with a low-dim random-projection
    prefilter (~64-dim) followed by exact dot products on the top-N candidates.

    See docs/06-mlx-lm-head-acceleration-plan.md in the LARQL repository for
    the full theory and design.
    """

    def __init__(
        self,
        weight: mx.array,        # [V, H], the original lm_head weight
        target_dim: int = 64,
        verify_n: int = 2048,
        seed: int = 0,
    ):
        super().__init__()
        V, H = weight.shape
        self.V = V
        self.H = H
        self.target_dim = target_dim
        self.verify_n = verify_n

        # Keep a reference to the full weight for verification.
        # MUST NOT copy — we want zero memory overhead beyond the projection.
        self.weight = weight  # [V, H]

        # Build the random projection deterministically.
        rng_key = mx.random.key(seed)
        scale = 1.0 / math.sqrt(target_dim)
        self.proj_matrix = mx.random.normal(
            shape=(target_dim, H), key=rng_key
        ) * scale  # [d, H]

        # Pre-project the weight: W @ P^T → [V, d]
        # Done once at construction time. Memory: V × d × 2 bytes (~32 MB at d=64, V=248K).
        self.weight_proj = (weight @ self.proj_matrix.T).astype(mx.float16)
        # mx.eval(self.weight_proj) here to materialize before first inference if eager
        # evaluation matters in MLX.

    def __call__(self, x: mx.array) -> mx.array:
        """
        x: [..., H] residual after final norm.
        Returns: [..., V] sparse logits — exact values at the top verify_n
                 candidates, -inf elsewhere. This is compatible with all
                 standard samplers (argmax, top-K, top-P, repetition penalty).
        """
        # Stage 1: approximate scores.
        x_proj = x @ self.proj_matrix.T              # [..., d]
        approx_scores = x_proj @ self.weight_proj.T  # [..., V]

        # Stage 2: pick top verify_n candidates and exact-compute their logits.
        # Use argpartition for O(V log N) top-N selection.
        # Note: MLX may not have argpartition; use sort + slice or top_k op.
        candidate_idx = mx.argpartition(-approx_scores, kth=self.verify_n, axis=-1)
        candidate_idx = candidate_idx[..., :self.verify_n]  # [..., verify_n]

        # Gather the candidate weight rows and compute exact dot products.
        # Shape gymnastics: weight is [V, H], we want [..., verify_n, H].
        candidate_weights = self.weight[candidate_idx]      # [..., verify_n, H]
        exact_scores = mx.einsum(
            "...h,...kh->...k", x, candidate_weights
        )                                                    # [..., verify_n]

        # Build sparse logit output: -inf everywhere except verified candidates.
        out = mx.full(approx_scores.shape, -mx.inf, dtype=x.dtype)
        # Scatter exact_scores into out at candidate_idx positions.
        # MLX's scatter API should be checked here — exact name may differ.
        out = mx.put_along_axis(out, candidate_idx, exact_scores, axis=-1)

        return out
```

Important details to verify against MLX's actual API:
- `mx.random.normal` with explicit `key` for reproducibility — confirm signature.
- `mx.argpartition` — may not exist; if not, use `mx.argsort(...)[:, :N]` (slower but correct).
- `mx.put_along_axis` — confirm name (could be `mx.scatter`, `mx.array.at[].set()`, or a manual approach).
- `mx.einsum` — confirm batch dims work as expected.

If MLX requires eager evaluation in places, you may need `mx.eval(...)` calls. Profile the first version and fix as issues appear.

### 4.2 Integration into the Qwen3.5 model class

The exact location of `lm_head` in `mlx-lm`'s Qwen3.5 model class will depend on how MLX-LM organizes models. Typical structure:

```python
# mlx_lm/models/qwen3_5.py (or similar — confirm at sprint start)
class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model = TransformerLayers(args)
        if args.tie_word_embeddings:
            # tied embedding case: lm_head is implicit
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                args.hidden_size, args.vocab_size, bias=False
            )

    def __call__(self, inputs, ...):
        out = self.model(inputs, ...)
        if self.lm_head is None:
            out = self.model.embed_tokens.as_linear(out)  # tied case
        else:
            out = self.lm_head(out)
        return out
```

Two integration approaches:

**Approach A (preferred): wrap lm_head after model construction.**

```python
def enable_approx_lm_head(model, target_dim=64, verify_n=2048):
    if model.lm_head is not None:
        approx = ApproxLMHead(
            weight=model.lm_head.weight,
            target_dim=target_dim,
            verify_n=verify_n,
        )
        model.lm_head = approx
    else:
        # Tied embedding case: wrap the embedding layer's as_linear
        # This is more invasive; defer to Approach B for tied embedding models.
        raise NotImplementedError("Tied embeddings: see Approach B")
```

Called after `load(...)` returns a model. Opt-in, additive, doesn't change model loading.

**Approach B: thread a flag through model loader.**

Add `approx_lm_head: bool = False` and `approx_lm_head_target_dim`, `approx_lm_head_verify_n` to the loader's argument list. Inside the loader, after weights are loaded, replace `lm_head` with `ApproxLMHead`. More invasive but cleaner from a configuration standpoint.

For the initial PR, **Approach A is simpler and easier to review.** The user can call `enable_approx_lm_head(model)` after loading. Approach B is a follow-up enhancement.

### 4.3 Sampling compatibility

The `__call__` method returns logits with the same shape as the dense baseline (`[..., vocab_size]`), but with `-inf` at positions outside the top-N candidates and exact values at the candidate positions. This is **deliberately compatible with every standard sampler**:

- **Argmax** — returns the candidate with the highest exact score.
- **Top-K** — softmax over the candidates (since `-inf` positions softmax to 0). Works as long as `K ≤ verify_n`.
- **Top-P** — same. The cumulative probability mass is supplied by the candidates; the rest of the vocab has 0 probability.
- **Temperature scaling** — `-inf / T = -inf`, no issue.
- **Repetition penalty** — modifies logits at specific past-token IDs. If those IDs are in the candidate set, the penalty applies; if not, they're already at `-inf` and the penalty is moot. Behaviorally equivalent to the dense case for any token whose dense logit would have been below the top-N cutoff.

**Edge case to verify:** repetition penalty assumes the penalized tokens are reachable. If a recent token has been pushed out of the candidate set by the approximation, the penalty has no effect. In practice this is fine because repetition is usually for high-probability tokens that *are* in the top-N. Document this caveat in the PR.

### 4.4 The MTP head

Qwen3.5 has a Multi-Token Prediction head used for speculative-decoding-style multi-token emission. It's separate from the main `lm_head`. For the initial PR:

- **Don't touch MTP.** The main lm_head is the dominant cost; MTP is a smaller secondary head.
- **Verify MTP still works** in the integration test (it should — we're only replacing the main lm_head).
- **Follow-up:** apply the same technique to MTP after the main lm_head ships.

If you find that MTP and the main lm_head share weights or have an entanglement, surface this and ask before proceeding.

---

## 5. Step-by-step implementation plan

### Phase 1: Environment + exploration (Day 1)

1. **Clone MLX-LM** to a working directory:
   ```bash
   git clone https://github.com/ml-explore/mlx-lm
   cd mlx-lm
   pip install -e .
   ```
   Verify the development install by running an existing example.

2. **Pull Qwen3.5-9B** (small variant for fast iteration):
   ```bash
   # Via mlx-lm's loader, or huggingface-cli download
   ```
   Confirm the model loads and produces sensible output:
   ```python
   from mlx_lm import load, generate
   model, tokenizer = load("Qwen/Qwen3.5-9B-MLX")
   print(generate(model, tokenizer, prompt="The capital of France is", max_tokens=10))
   ```

3. **Locate the Qwen3.5 model class.** Search `mlx_lm/models/` for `qwen3_5*` or `qwen3*`. Read the model class. Identify:
   - Where `lm_head` is defined (its name, type, weight shape).
   - Where it's called in the forward pass.
   - Whether `tie_word_embeddings` is true or false for the variant under test.
   - How the model receives configuration parameters.

4. **Run a simple baseline benchmark.** Generate, say, 200 tokens at temperature 0 (greedy) with a fixed prompt, measure tokens/second. Record this as the baseline. Use:
   ```bash
   mlx_lm.generate --model Qwen/Qwen3.5-9B-MLX --prompt "..." --temp 0 --max-tokens 200
   ```
   or equivalent. Note the tok/s number.

### Phase 2: Core implementation (Days 2-3)

5. **Create the `ApproxLMHead` module.** New file: `mlx_lm/utils/approx_lm_head.py`. Implement as in §4.1 above. Pay close attention to:
   - MLX's exact API for `argpartition` / `top_k` / `argsort`.
   - MLX's scatter / `put_along_axis` semantics.
   - Whether MLX needs explicit `mx.eval()` calls for materialization.
   - Whether the `weight @ P.T` projection happens lazily or eagerly (we want eagerly, at construction).

6. **Write the `enable_approx_lm_head` helper.** Approach A from §4.2. Should be a few lines: replace `model.lm_head` with `ApproxLMHead(weight=model.lm_head.weight, ...)`.

7. **Smoke test.** With Qwen3.5-9B loaded and the helper applied, generate a token and confirm it's reasonable. Compare greedy outputs across 10 prompts: with `verify_n` large enough (e.g. 8192), greedy outputs should be **identical** to dense.

8. **Profile**: how long does load-time projection take? Measure `enable_approx_lm_head` execution. Confirm it's < 30 seconds on Qwen3.5-9B and < 5 minutes on the 122B variant.

### Phase 3: Validation (Days 3-5)

9. **Recall sweep.** Write a script that:
   - Loads dense and approx versions of the same model.
   - Generates 1000 prompts (mix of chat / code / multilingual).
   - For each prompt, runs forward to the final residual; computes both dense and approx logits.
   - Computes jaccard(top-50_dense, top-50_approx) for each prompt.
   - Aggregates: mean, p50, p95, min jaccard.
   - **Target: mean ≥ 0.99, p95 ≥ 0.97, min ≥ 0.90** at default settings (target_dim=64, verify_n=2048).

10. **Quality eval.** Pick 1-2 small evaluation suites:
    - **MMLU subset** (1000 random questions): measure accuracy with both dense and approx.
    - **HellaSwag subset** (500 questions): same.
    - Acceptable drift: ≤ 0.5% absolute.
    
    These exist in `lm-eval-harness`-compatible form; if MLX-LM has its own eval scaffold, use it. If not, write a minimal evaluator.

11. **Greedy parity check.** Generate 200 tokens for each of 50 diverse prompts at temperature=0. Compare token-by-token against dense baseline. **Target: ≥ 99% of tokens identical** (any divergence will mostly be at numerically tied positions, which random projection may break differently than dense).

12. **Tune knobs if needed.** If recall or quality targets aren't met:
    - First, increase `target_dim` from 64 → 128. Doubles stage-1 cost (still negligible) but materially improves recall.
    - Then, increase `verify_n` from 2048 → 4096. Doubles stage-2 cost.
    - If still failing: switch from random projection to PCA (top-k right singular vectors of `lm_head`). One-time SVD, ~30-60 seconds at load time, better recall per `target_dim`.
    - Document the final settings in the PR.

### Phase 4: Scale up (Days 5-7)

13. **Run on Qwen3.5-122B-A10B.** This requires sufficient hardware (Mac Studio M3 Ultra with 192+ GB recommended). Repeat:
    - Load-time benchmark (projection build cost).
    - Smoke test (generates sensibly).
    - Greedy parity (100 prompts).
    - Speed benchmark: tok/s on a fixed prompt set, before vs after.
    - Memory benchmark: RSS before and after the projection is built.

14. **Speed measurement methodology.** Use a long enough output (≥ 200 tokens) to amortize prefill costs. Run 5 trials, report median. Compare:
    - Baseline: dense lm_head.
    - Approx: ApproxLMHead with default settings.
    - Approx with `verify_n=4096` (higher quality, smaller speedup).
    
    **Target: ≥ 8% total decode speedup at default settings on the 122B variant.** Realistic range: 8-15%.

15. **(Optional stretch) Validate on Qwen3.5-397B-A17B at Q4_K_M** if hardware is available. Same suite. Speed improvement may be smaller in proportion (bigger model → MoE expert dispatch dominates more) but should still be measurable.

### Phase 5: PR (Days 7-10)

16. **Write tests.** At minimum:
    - Unit test: `ApproxLMHead` forward produces the same shape as dense.
    - Unit test: at very large `verify_n`, output equals dense within numerical tolerance.
    - Unit test: at small `verify_n`, top-1 still matches dense for common prompts.
    - Integration test: `enable_approx_lm_head` + generation produces valid output.

17. **Documentation.** Add to MLX-LM's docs / README:
    - What the feature does.
    - When to enable it (large-vocab models, large MoE models).
    - Expected speedup numbers from your measurements.
    - Configuration knobs (`target_dim`, `verify_n`).
    - Known limitations (the repetition penalty caveat).

18. **Benchmark script.** Include a `examples/bench_approx_lm_head.py` that reproduces your speed measurements. Important for review: maintainers want to validate the numbers themselves.

19. **Open the PR.** Title: "feat: ApproxLMHead — random-projection acceleration for large-vocab models." Body:
    - One-paragraph summary of the optimization and the speedup.
    - The theory link (this document or a condensed version).
    - The benchmark numbers (with hardware spec).
    - The recall/quality validation.
    - The opt-in mechanism.
    - List of models tested.
    
    Keep it factual and measured; large-PR success comes from clean numbers and minimal claims, not enthusiasm.

20. **Iterate on review.** Expect questions about:
    - Repetition penalty interaction (have an answer ready).
    - Whether MTP is affected (no, but be ready to demonstrate).
    - Recall at extreme settings (have measurements at multiple `verify_n` values).
    - Whether the projection is stable across MLX runs (it is if seeded).

---

## 6. Validation criteria (definition of done)

A green sprint requires all of:

| Criterion | Target | How measured |
|---|---|---|
| Top-50 jaccard recall (mean) | ≥ 0.99 | Recall sweep on 1000 prompts |
| Top-50 jaccard recall (p95) | ≥ 0.97 | Same |
| MMLU accuracy drift | ≤ 0.5% absolute | 1000-question MMLU subset |
| HellaSwag accuracy drift | ≤ 0.5% absolute | 500-question subset |
| Greedy parity rate | ≥ 99% | 50 prompts × 200 tokens = 10K positions |
| Total decode speedup (Qwen3.5-122B-A10B) | ≥ 8% | 5-trial median, 200-token greedy |
| Load-time projection cost | ≤ 5 minutes on 122B | Wall clock |
| Memory overhead | ≤ 5% of model RAM | RSS measurement |
| Tests pass | 100% | CI |

If any criterion is not met after tuning, document the gap in the PR honestly. A 5% speedup with 99.5% recall is still a worthwhile PR if presented accurately.

---

## 7. Reference materials

### 7.1 Internal docs (this LARQL repository)

- [`docs/01-high-level-explainer.md`](01-high-level-explainer.md) — LARQL conceptual overview. The graph view of FFN.
- [`docs/02-mathematical-foundations.md`](02-mathematical-foundations.md) — **directly relevant**. §2 (matmul-as-graph), §3 (sparsity bounds), §4 (gate KNN as MIPS) ground the same math we're applying to `lm_head` here.
- [`docs/03-software-architecture.md`](03-software-architecture.md) — how LARQL organizes a similar primitive (gate KNN dispatch with multiple backends).
- [`docs/05-qa-and-toy-implementation.md`](05-qa-and-toy-implementation.md) — discusses the KNN-based logit projection approach in §Q1 with similar math.
- [`docs/walk-boundary-sweep.md`](walk-boundary-sweep.md) — empirical proof that the approximate-then-verify pattern preserves output. Gemma 3 4B, all 34 layers, zero divergence.
- [`docs/ffn-graph-layer.md`](ffn-graph-layer.md) — engineering details of how LARQL implemented sparse FFN.

### 7.2 External references

- **Johnson-Lindenstrauss lemma:** original is Johnson & Lindenstrauss 1984. A clean modern treatment with the variant we're using: Achlioptas 2003, "Database-friendly random projections."
- **Random projection for nearest-neighbor search:** Indyk & Motwani 1998, "Approximate nearest neighbors: towards removing the curse of dimensionality."
- **FAISS** (for orientation): <https://github.com/facebookresearch/faiss>. Demonstrates the production engineering of approximate top-K. We are using the simplest possible variant of what FAISS contains.
- **MLX core docs:** <https://ml-explore.github.io/mlx/build/html/index.html>. Reference for any API question.
- **MLX-LM repo:** <https://github.com/ml-explore/mlx-lm>. The target.
- **Qwen3-Next paper / blog:** <https://qwen3-next.com/>. Background on the architecture if needed.

### 7.3 Code touchpoints in MLX-LM

To be confirmed at sprint start, but expected:

- `mlx_lm/models/qwen3_5.py` (or the canonical Qwen3.5 model file). Where `lm_head` is defined and called.
- `mlx_lm/utils/__init__.py` or similar. Where `enable_approx_lm_head` will live.
- `mlx_lm/sample_utils.py` or `mlx_lm/generate.py`. The sampler integration point — verify that samplers tolerate `-inf` logits (they should).
- `tests/`. Where unit tests live.
- `examples/` or `benchmarks/`. Where the speed benchmark will live.

---

## 8. Pitfalls and watch-outs

Things that have bitten similar engineering work in the past. Read this before you start.

### 8.1 MLX lazy evaluation

MLX has lazy evaluation by default. Operations queue up; results materialize on `mx.eval()` or when crossing a Python boundary (e.g. converting to numpy, printing, indexing). This can mask performance bugs:

- The `weight @ P.T` projection looks instant if MLX hasn't actually computed it yet.
- The "speedup" you think you're measuring may be benchmark artifact if you don't `mx.eval()` at the right spots.

**Always `mx.eval()` after the projection build and before measuring inference time.** Profile with `mx.metal.start_capture()` if available to be sure compute is happening when you think it is.

### 8.2 Tied vs untied embeddings

Most modern models have tied word embeddings: `lm_head.weight = embed_tokens.weight`. Qwen3.5 uses untied embeddings for the larger MoE variants (confirm at sprint start), but the dense 9B might use tied.

If the model uses tied embeddings, replacing `lm_head` may break the embedding lookup. Handle by:
- Detecting the tied case (`model.lm_head is None` or similar).
- Wrapping at a higher level (the model's `__call__` method) rather than at `lm_head`.
- Or: documenting that the feature requires untied embeddings and noting which Qwen3.5 variants are compatible.

### 8.3 Quantized weights

If the model is loaded with quantization (Q4 or NVFP4), `model.lm_head.weight` may not be a plain f16/f32 MLX array — it could be a quantized representation. The projection `weight @ P.T` may not work directly.

Handle by:
- Detecting quantization at construction time.
- Dequantizing the lm_head weight to f16 once at projection build time (this is already what the GPU kernel does on every token).
- The projected `W_proj` is small (~32 MB) and can be kept at f16 regardless.

The `weight[candidate_idx]` gather in the verify step also needs to handle quantized weights — likely by using whatever dequant kernel MLX-LM already uses. Check the existing `nn.Linear` quantized path for the right API.

### 8.4 First-token vs subsequent-token cost

Generation has two phases: prefill (process the entire prompt once) and decode (one token at a time). Prefill batches many tokens through lm_head at once, which actually reduces relative bandwidth pressure (you read lm_head once, use it for many tokens of output).

**The optimization helps decode much more than prefill.** Make sure your benchmark measures decode (typically tok/s after the first token) not prefill. Most MLX-LM benchmarks already do this correctly.

### 8.5 Numerical stability

`-inf` arithmetic is well-behaved in IEEE float (subtraction works, addition works, multiplication by positive number works). But `0 × -inf = NaN`. Confirm no sampler does `mask * logits` where `mask` could be 0 at a `-inf` position.

### 8.6 Don't break existing models

Your changes should be entirely additive. If `enable_approx_lm_head` is never called, the behavior of MLX-LM should be byte-identical to before your PR. Maintain this discipline ruthlessly during the sprint — it's the property that makes the PR easy to merge.

### 8.7 Tokenizer / vocab edge cases

Some token IDs may be reserved (padding, BOS, EOS, special tokens). The lm_head still has rows for them; the dense baseline produces logits at those positions and samplers handle them. Make sure your top-N candidate set can include them, and that the sampler's special-token handling still works.

### 8.8 Don't redesign without authorization

The user has approved the random-projection approach specifically. If you discover a fundamental blocker (e.g. MLX won't let you do the projection efficiently), surface it and ask. **Do not silently switch to HNSW, FAISS, or another algorithm.** The user evaluated alternatives and chose this one for specific reasons (no sidecars, no dependencies, instant build).

---

## 9. Communication and coordination

- **The user (project owner):** this plan is approved. They will provide hardware access and any credentials needed for HuggingFace model downloads.
- **MLX-LM maintainers:** the user has not pre-coordinated with them. Open a discussion issue *before* the PR if the change touches more than the model file + a new utility. A short heads-up issue ("planning to add ApproxLMHead for Qwen3.5; here's the math; OK to proceed?") is generally well-received.
- **You (the agent):** keep the user updated via short status messages at the end of each phase. Don't ask for permission on within-plan decisions; do ask for guidance on out-of-plan choices.
- **LARQL repository owner:** Chris Hayuk. The user will coordinate any cross-references back to this repo (this plan document, citations).

---

## 10. Beyond this sprint (future work, out of scope)

For context, after this PR lands cleanly, natural follow-ups in priority order:

1. **Port to llama.cpp.** Same algorithm, C++ implementation. Larger user base, harder PR. Wait until MLX version has 1-2 months of production validation.
2. **Apply to gate KNN (per layer FFN selection).** Same technique, applied to FFN gate vectors. The win is smaller per-layer but applies to every FFN layer, so it compounds. Requires more invasive engine changes.
3. **PCA prefilter as an alternative.** For maximum recall at the same target_dim, swap random projection for the top-k right singular vectors of `lm_head`. Higher build cost, slightly better recall.
4. **Sidecar mode for production deployment.** Once the technique is proven, allow saving the projection matrix to disk for instant load (avoiding even the 5-second build). Sidecar pattern as discussed in earlier planning.
5. **MTP head extension.** Apply ApproxLMHead to Qwen3.5's MTP head as well.
6. **Larger-scale validation.** Run on Kimi K2, GLM-4.5/4.6, DeepSeek V3 to confirm the technique works on all large-vocab models.

None of these are in scope for the initial sprint. **Focus on shipping the simple version cleanly.** The follow-ups become much easier to argue for once the simple version is in production.

---

## 11. One-line summary

> Replace Qwen3.5's `lm_head` projection in MLX-LM with a two-stage approximate-then-exact search: a 64-dim Gaussian random-projection prefilter on `[V × H]` followed by exact dot products on the top-2048 candidates. Built at load time in seconds, no sidecar files. Mathematically grounded in Johnson-Lindenstrauss; engineering-grounded in LARQL's gate-KNN pattern. Expected: ~8-12% total decode speedup on Qwen3.5-122B-A10B with negligible quality loss.

Good luck. The math is sound, the engineering is bounded, and the PR target is well-defined. Ship cleanly.
