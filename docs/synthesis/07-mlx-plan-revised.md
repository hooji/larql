# MLX LM Head Acceleration — Revised Plan (PCA-Primary, Tied-Embed-Aware)

> Revised version of the original `06-mlx-plan-original` random-projection
> sprint plan. Same target (a 30%-tier `lm_head` speedup in `mlx-lm` for
> Qwen 3.5 family and other large-vocab models), same shape, same opt-in
> defaults, same validation discipline — but with engineering corrections
> identified in [`06-mlx-plan-review.md`](06-mlx-plan-review.md). The
> review document explains each change; this is the executable plan.

---

## 0. For the agent picking this up

Welcome. This is a refined sprint plan replacing
`06-mlx-plan-original.md`. **Read this entire document first**, then
the review at `06-mlx-plan-review.md` for the rationale behind the
changes. Then read these supporting docs in this order:

1. [`docs/synthesis/01-explainer.md`](01-explainer.md) — LARQL conceptual frame.
2. [`docs/synthesis/02-mathematical-foundation.md`](02-mathematical-foundation.md) — sections 2–4 directly apply: matmul-as-graph, gate-KNN as MIPS, sparsity bounds.
3. [`docs/synthesis/03-software-architecture.md`](03-software-architecture.md) — how LARQL organises the analogous primitive.
4. [`docs/walk-boundary-sweep.md`](../walk-boundary-sweep.md) — empirical proof that "sparse top-K dispatch" preserves output.

The user has approved this plan. **You are explicitly authorised to use
either Random Projection or PCA** as the prefilter (see §4.3). Other
algorithm changes (HNSW, FAISS, product quantisation) require explicit
approval — surface and ask if you find a fundamental blocker.

---

## 1. Executive summary

**What we're building.** A drop-in approximation for the language-modeling
head (`lm_head`) projection in `mlx-lm` for the Qwen 3.5 family. The dense
`[V × H]` matmul that produces logits is replaced with a two-stage
approximate-then-verify search: a low-dimensional pre-filter selects ~2,048
candidate tokens, then exact dot products run only on those candidates.
The pre-filter is built at model load time using either truncated
randomised PCA (default, better recall) or a Gaussian random projection
(fallback). No sidecar files. No external dependencies.

**Why it matters.** On Qwen 3.5-122B-A10B (vocab=248,320, hidden≈5,120),
the dense `lm_head` is ~2.5 GB of weight read per token — a memory-bandwidth
bottleneck on Mac. Sampling needs only the top of the distribution; the
rest is wasted bandwidth. Replacing the dense gemv with a 64-dim
prefilter + exact verify on top-2,048 candidates reduces effective
`lm_head` cost by ~5–10× (after accounting for gather inefficiency and
kernel-launch overhead, *not* the headline 47× from naive bandwidth math).
Realistic translation: **~5% total decode speedup** (target), with 8% as
a stretch goal.

**Why now.** Qwen 3.5's hybrid attention (Gated DeltaNet + GQA, 75/25 mix)
and ultra-sparse MoE (top-10 of 512 experts) push proportionally more decode
cost into the dense ends of the model — `lm_head` and embeddings — making
this the most valuable single optimisation for this architecture today,
and the same pattern will apply to DeepSeek V3/V4, Kimi K2/K2.6, and
GLM-4.5/4.6 after.

**Constraints.**
- No sidecar files. Index built at load time.
- Pure MLX core, no external deps (no hnswlib, faiss).
- Greedy output identical to dense. Top-K / top-P distributions
  statistically indistinguishable.
- Opt-in via helper function; default off until validated.
- Existing samplers (greedy, top-K, top-P, repetition penalty) must
  work without modification.

**Definition of done.**
- A PR against `ml-explore/mlx-lm` (preceded by a discussion issue).
- Unit + integration tests.
- Benchmarks on Qwen 3.5-9B (dev) and Qwen 3.5-122B-A10B (target).
- Recall + quality validation: see §6.
- PR description references this plan and the LARQL math/architecture docs.

---

## 2. Theoretical background

(Unchanged from the original plan in mathematical content; see
`06-mlx-plan-original.md` §2 for the full treatment. Summary below.)

### 2.1 The lm_head bottleneck

Per-token bandwidth read for `lm_head` on Qwen 3.5-122B-A10B is ~2.5 GB.
On Mac Studio M3 Ultra (~800 GB/s effective unified-memory bandwidth on
GPU), that's ~3 ms per token *just for lm_head*. As a fraction of total
decode time on this MoE model — which only reads ~10 B active params per
token (~20 GB read) — `lm_head` is **~10–12% of decode bandwidth**. That
is what's available to be optimised away.

### 2.2 What sampling needs

Greedy: top-1. Top-K (K≤200): top-K. Top-P: top few hundred. Repetition
penalty: a few specific past-token IDs, plus the rest of the
distribution shape. None of these need the bottom 99%.

### 2.3 Two-stage approximate-then-verify

```
# Build phase (once, at model load):
P ∈ ℝ^{d × H}                            ← PCA top-d (or random Gaussian)
W_proj ∈ ℝ^{V × d}                       ← W @ P^T, materialised

# Query phase (per token):
q_proj = q @ P^T                         ∈ ℝ^d            (negligible cost)
approx = q_proj @ W_proj^T               ∈ ℝ^V            (V × d ops)
cand   = top_N(approx)                   ∈ ℤ^N            (N ≈ 2048)
exact  = W[cand] @ q                     ∈ ℝ^N            (N × H ops, gather)
return scatter(exact at cand into [-inf]^V)
```

### 2.4 Realistic cost analysis (corrected)

Cost on Qwen 3.5-122B-A10B (V=248,320, H=5,120, d=64, N=2,048):

| Stage | Ops | Bandwidth (f16) | Realistic Metal time |
|---|---|---|---|
| Stage 1 query proj | 0.33 M | <1 MB | <50 µs |
| Stage 1 V × d matmul | 15.9 M | ~32 MB | ~150–250 µs (tall-skinny shape) |
| Stage 1 top-N partition | — | — | ~100–300 µs |
| Stage 2 random-row gather | — | ~21 MB random | ~100–200 µs (gather, not contiguous) |
| Stage 2 N × H matmul | 10.5 M | (above) | ~50–100 µs |
| Stage 2 scatter to [V] | — | ~500 KB write | ~30 µs |
| **Total** | **~26 M** | **~54 MB nominal** | **~430–930 µs** |
| Dense baseline | 1.27 B | 2.5 GB contiguous | ~3–6 ms |

The realistic improvement on `lm_head` itself is **5–10× wall-clock**, not
the 47× suggested by naive bandwidth math. Translating to total decode
speedup (where `lm_head` is ~10% of bandwidth): **~5% speedup** is the
target, ~8% is the stretch goal, ~10% is the absolute ceiling assuming
everything goes well and we recover Stage 2 gather efficiency through
pinning or kernel-fusion.

### 2.5 Connection to LARQL gate-KNN

Same pattern. The walk-boundary sweep on Gemma 3-4B
([`docs/walk-boundary-sweep.md`](../walk-boundary-sweep.md)) demonstrates
that this approach preserves output bit-near-perfectly across all 34
layers. We are doing the same thing for `lm_head` — one op, simpler
integration.

---

## 3. Target environment

Qwen 3.5 family (as of May 2026):
- Qwen 3.5-9B (dense, vocab=248,320, hidden=4,096, layers=32)
- Qwen 3.5-27B (dense)
- Qwen 3.5-35B-A3B (MoE, 35B/3B active)
- Qwen 3.5-122B-A10B (MoE, 122B/10B active, hidden≈5,120) — primary target
- Qwen 3.5-397B-A17B (MoE, 397B/17B active) — **deferred** to follow-up

Hybrid attention (75% Gated DeltaNet + 25% Gated GQA) and ultra-sparse
MoE (top-10/512 experts) do not affect `lm_head` directly, but make
`lm_head` proportionally more dominant in decode bandwidth.

**Multi-Token Prediction head:** Qwen 3.5's MTP is used for
self-speculative decoding. The main `lm_head` is what we optimise; the
MTP head is unchanged. **However** — the interaction between approximate
main-`lm_head` and speculative verification must be measured. See §5
Phase 3.

**Tied vs untied embeddings:**
- Qwen 3.5-9B (dev target): **likely tied** — must be supported in v1.
- Qwen 3.5-122B-A10B (perf target): likely untied. Confirm at sprint start.

**Engine.** `mlx-lm` (`ml-explore/mlx-lm`). Same rationale as original
plan: Python iteration speed, Mac Silicon native, the team has shipped
Qwen-specific optimisations before, llama.cpp is a worthy follow-up
target *after* MLX validates.

---

## 4. Implementation design

### 4.1 The `ApproxLMHead` module

```python
import math
import mlx.core as mx
import mlx.nn as nn

class ApproxLMHead(nn.Module):
    """Two-stage approximate-then-verify language-modeling head.

    Replaces a dense [V, H] projection with a low-dim prefilter
    (PCA-default or random-projection-fallback) plus exact dot products
    on the top-N candidates.

    See docs/synthesis/07-mlx-plan-revised.md for theory and design.
    """

    def __init__(
        self,
        weight: mx.array,        # [V, H], the original lm_head weight
        target_dim: int = 64,
        verify_n: int = 2048,
        method: str = "pca",     # "pca" (default) or "rp"
        seed: int = 0,           # only used for method="rp"
    ):
        super().__init__()
        V, H = weight.shape
        self.V = V
        self.H = H
        self.target_dim = target_dim
        self.verify_n = verify_n
        self.method = method

        # Reference (not copy) to original weight for verify-stage gather.
        self.weight = weight  # [V, H]

        # Build the projection matrix P ∈ [d, H].
        if method == "pca":
            # Truncated randomised SVD on weight; P = top target_dim right
            # singular vectors. Gives empirically ~2-4x better recall than RP.
            self.proj_matrix = self._build_pca_projection(weight, target_dim)
        elif method == "rp":
            rng_key = mx.random.key(seed)
            scale = 1.0 / math.sqrt(target_dim)
            self.proj_matrix = mx.random.normal(
                shape=(target_dim, H), key=rng_key
            ) * scale
        else:
            raise ValueError(f"unknown method: {method}")

        # Pre-project the weight: W @ P^T → [V, d]
        # MUST materialise eagerly — MLX is lazy by default and would
        # otherwise rebuild the projection on every forward pass.
        self.weight_proj = (weight @ self.proj_matrix.T).astype(mx.float16)
        mx.eval(self.weight_proj, self.proj_matrix)  # FORCE materialisation

        # Reusable -inf output buffer; avoid per-token allocation.
        # Shape [V] in f32 (so softmax handles -inf cleanly even with f16 in).
        self._inf_buffer = mx.full((V,), -mx.inf, dtype=mx.float32)
        mx.eval(self._inf_buffer)

    def _build_pca_projection(self, weight: mx.array, d: int) -> mx.array:
        """Top-d right singular vectors of weight via randomised SVD.

        Uses the standard Halko-Martinsson-Tropp 2011 algorithm with
        oversampling factor 10 and 2 power iterations. Build cost ~10-30s
        on a 248K x 5120 matrix on Metal.
        """
        V, H = weight.shape
        oversample = 10
        n_iter = 2
        # Random sketching matrix
        omega = mx.random.normal(shape=(H, d + oversample))
        # Power iteration for spectral concentration
        Y = weight @ omega                          # [V, d+os]
        for _ in range(n_iter):
            Y = weight @ (weight.T @ Y)             # cheap iterations
        # Orthonormalise
        Q, _ = mx.linalg.qr(Y, stream=mx.cpu)       # [V, d+os]
        # Smaller matmul + SVD
        B = Q.T @ weight                            # [d+os, H]
        U, S, Vt = mx.linalg.svd(B, stream=mx.cpu)  # full SVD on small matrix
        # Top-d right singular vectors
        return Vt[:d]                               # [d, H]

    def __call__(self, x: mx.array) -> mx.array:
        """
        x: [..., H] residual after final norm.
        Returns: [..., V] sparse logits — exact at top verify_n positions,
                 -inf elsewhere. Compatible with all standard samplers.
        """
        # Stage 1: approximate scores
        x_proj = x @ self.proj_matrix.T               # [..., d]
        approx_scores = x_proj @ self.weight_proj.T   # [..., V]

        # Stage 2: pick top verify_n candidates
        # mx.argpartition is available; if not on your MLX version,
        # fall back to mx.topk (which exists per current MLX docs).
        cand = mx.argpartition(-approx_scores, kth=self.verify_n, axis=-1)
        cand = cand[..., :self.verify_n]              # [..., N]

        # Gather candidate weight rows
        cand_weights = mx.take(self.weight, cand, axis=0)   # [..., N, H]
        exact_scores = mx.einsum("...h,...kh->...k", x, cand_weights)

        # Sparse output: clone the -inf buffer (cheap), scatter exact_scores.
        # The clone+scatter pattern is the price of sampler interface compat;
        # buffer reuse keeps it from being a fresh allocation each token.
        out = mx.broadcast_to(self._inf_buffer, approx_scores.shape).copy()
        out = mx.put_along_axis(out, cand, exact_scores.astype(mx.float32), axis=-1)
        return out
```

Notes on changes from the original plan:
- **PCA primary, RP fallback** via `method=` parameter.
- **Eager materialisation enforced** with explicit `mx.eval(...)`.
- **`_inf_buffer` reused** across tokens, not per-token allocated.
- **Output dtype is f32** even when input is f16, so `-inf` survives
  softmax cleanly.
- **`mx.take` instead of fancy indexing** for the row gather, since
  `mx.take` has a more predictable Metal kernel path.

### 4.2 Tied embeddings — first-class case

```python
def enable_approx_lm_head(
    model,
    target_dim: int = 64,
    verify_n: int = 2048,
    method: str = "pca",
    seed: int = 0,
):
    """Wrap a model's lm_head with ApproxLMHead. Works for tied OR untied."""
    untied = getattr(model, "lm_head", None) is not None

    if untied:
        weight = model.lm_head.weight
        approx = ApproxLMHead(weight, target_dim, verify_n, method, seed)
        model.lm_head = approx
        return

    # Tied case: weight is in embed_tokens
    embed = getattr(model, "embed_tokens", None) or model.model.embed_tokens
    weight = embed.weight
    approx = ApproxLMHead(weight, target_dim, verify_n, method, seed)
    # Install as the model's effective lm_head and patch the forward path.
    # The exact patching mechanism depends on mlx-lm's model structure;
    # the typical Qwen3 model already has a "head" call site that uses
    # `model.embed_tokens.as_linear(out)` for tied — replace that path.
    model._approx_lm_head = approx
    _patch_tied_forward(model)


def _patch_tied_forward(model):
    """For tied-embedding models, redirect the as_linear call to use
    ApproxLMHead. Implementation depends on the mlx-lm Qwen 3.5 model
    class; locate the call site at sprint start."""
    # Pseudocode — fill in based on actual model structure:
    #   - Find the `__call__` (or named) method that does
    #     `self.model.embed_tokens.as_linear(out)`.
    #   - Replace with `self._approx_lm_head(out)`.
    raise NotImplementedError("Implement at Phase 2 step 6")
```

The tied case must be working before we benchmark on Qwen 3.5-9B,
because Qwen 3.5-9B is likely tied. Validate on a tied checkpoint
first, then on an untied checkpoint.

### 4.3 Algorithm choice (PCA vs RP) — explicit policy

| Method | Build cost (122B) | Memory | Recall@d=64 | When to use |
|---|---|---|---|---|
| **PCA (default)** | 10–30 s | 32 MB | top-50 jaccard ≥ 0.99 | Always, unless build cost is prohibitive |
| RP (fallback) | <1 s | 32 MB | top-50 jaccard ≈ 0.95–0.99 | Constrained build time; reproducibility from seed only |

The plan **explicitly authorises** the agent to use PCA for the default
implementation. Both methods produce a `[d × H]` projection matrix and
the rest of the pipeline is identical, so the choice is encapsulated
inside `ApproxLMHead.__init__` and doesn't affect anything downstream.

Two empirical knobs:
- `target_dim`: try 32, 64, 128 in validation. PCA at 32 may already
  hit the recall target.
- `verify_n`: try 1024, 2048, 4096. Smaller is faster but less safe.

### 4.4 Sampling compatibility

Same interface as original plan: `[..., V]` with `-inf` outside the
candidates is compatible with greedy, top-K, top-P, temperature, and
repetition penalty.

Repetition-penalty caveat (unchanged): if a recently-emitted token
falls outside the top-N candidate set in a given step, the penalty has
no logit to modify. In practice this is fine because repetition
penalty targets are usually high-probability tokens that are *in* the
top-N. Document this in the PR.

### 4.5 MTP head — explicit handling

The MTP head is **not modified**, but its interaction with approximate
main-`lm_head` must be measured:

- Approximate main-`lm_head` produces a slightly different distribution
  at the verification position than dense.
- This changes the speculative-decoding accept rate.
- It may change generated tokens at temperature 0 if the accept/reject
  decision flips at borderline cases.

In Phase 3 we measure: *with MTP enabled*, does the sequence of accepted
draft tokens match between dense and approx? If accept rate drops by
> 5%, surface and discuss with the user. Possible mitigations include
running MTP verify with dense lm_head (defeating much of the speedup) or
accepting the modest accept-rate decline.

---

## 5. Step-by-step implementation plan

Two-week sprint, four phases. Same shape as original plan; corrections
in italics.

### Phase 1: Environment + exploration (Day 1)

1. **Clone MLX-LM** (`git clone https://github.com/ml-explore/mlx-lm`).
2. **Pull Qwen 3.5-9B**, confirm dense baseline generation works.
3. **Locate the Qwen 3.5 model class.** Identify:
   - `lm_head` definition and shape.
   - Whether 9B uses tied embeddings (`tie_word_embeddings`).
   - The forward-path call site for the head.
   - The model loader entry point.
4. **Run baseline benchmark.** *Median of 20 trials* (not 5),
   200-token greedy, fixed prompt, with `caffeinate -i` and warm-up.
   Record tok/s baseline + IQR.
5. *(NEW)* **Open a discussion issue** on `ml-explore/mlx-lm` describing
   the planned change, citing this plan, asking for early review on the
   integration approach. Wait for at least an acknowledgement before
   sinking large effort into the PR.

### Phase 2: Core implementation (Days 2–4)

6. **Implement `ApproxLMHead`** in `mlx_lm/utils/approx_lm_head.py`,
   using §4.1 as the reference. Implement PCA via randomised SVD;
   keep RP as a fallback path.
7. **Implement `enable_approx_lm_head`** including the *tied-embedding
   path* (§4.2). Implement `_patch_tied_forward` for Qwen 3.5-9B's
   actual model structure.
8. **Smoke test on Qwen 3.5-9B (tied).** Generate output with
   `verify_n=8192` (very generous) and confirm bit-identical greedy
   output to dense across 10 prompts. This is the easy correctness
   check.
9. **Smoke test on Qwen 3.5-122B-A10B (untied).** Generate output
   with default settings; confirm output is sensible. *(Don't try to
   hit the speedup target yet — just verify nothing's broken.)*
10. **Profile build-time projection cost.** PCA randomised SVD on
    122B should be 10–30 s. If it's >60 s, surface and ask whether to
    fall back to RP for the 122B variant.
11. *(NEW)* **Verify lazy-evaluation correctness.** After
    `enable_approx_lm_head`, run `mx.metal.start_capture()` (if
    available) on a single forward pass; confirm Stage 1 and Stage 2
    each show up as exactly one Metal kernel and that the projection
    matrix is *not* recomputed.

### Phase 3: Validation (Days 4–7)

12. **Recall sweep.** 1,000 prompts (chat / code / multilingual /
    long-context). Compute jaccard(top-50_dense, top-50_approx) per
    prompt. *Plus*: jaccard(top-10) — much harder than top-50, where
    sampling actually picks. *Plus*: KL(approx_softmax || dense_softmax)
    over the top-P=0.95 mass.
    - **Targets:** top-50 mean jaccard ≥ 0.99, p95 ≥ 0.97;
      top-10 mean jaccard ≥ 0.95;
      mean KL ≤ 0.01 nats.
13. **Quality eval.** MMLU subset (1,000 q) + HellaSwag (500 q).
    Acceptable drift ≤ 0.5% absolute. *Run quick versions (100 q each)
    after Phases 2 and 3 to catch regressions early.*
14. **Greedy parity.** 50 prompts × 200 tokens; ≥ 99% positions
    identical. (Demoted from "definition of done" to "sanity check"
    per review.)
15. *(NEW)* **MTP interaction test.** With Qwen 3.5-122B's MTP head
    enabled, run 50 prompts × 100 tokens. Compare:
    - Final token sequence at temp=0 (dense vs approx with MTP both on).
    - Speculative accept rate.
    - Tokens/second.
    
    **Pass:** accept rate within ±5% of dense, output identity ≥ 95%
    at temp=0, tok/s improvement positive (not negative — i.e. MTP
    interaction didn't *worsen* speed by reducing accept rate more
    than the lm_head saving).

16. **Tune knobs if any target missed.**
    - First, swap method=pca for rp (or vice versa) to see which is
      better on this specific weight matrix.
    - Then, increase `target_dim` 64 → 128 (negligible cost increase,
      meaningful recall gain).
    - Then, increase `verify_n` 2048 → 4096.
    - Document final settings in PR.

### Phase 4: Scale + benchmarks (Days 7–10)

17. **Final benchmark on Qwen 3.5-9B (tied dev target).**
    Median of 20 trials, 200-token greedy, with the methodology
    discipline from §5.4. Report tok/s before/after, IQR, and
    decode-only-fraction speedup.
18. **Final benchmark on Qwen 3.5-122B-A10B (untied perf target).**
    Same methodology. **Target: ≥ 5% total decode speedup;
    8% is stretch, 10% is ceiling.**
19. **Memory benchmark.** RSS before vs after `enable_approx_lm_head`
    on both variants. Should be ~33 MB increase. **Pass: < 50 MB
    overhead (or 0.1% of model RAM, whichever is larger).**
20. *(REMOVED)* Original plan's "stretch: validate on 397B" — defer
    to a follow-up after the 122B PR lands. Don't burn sprint hours
    on hardware-availability issues for a stretch goal.

### Phase 5: PR (Days 10–14)

21. **Tests.**
    - Unit: forward shape correctness.
    - Unit: at very large `verify_n`, output ≈ dense within
      numerical tolerance.
    - Unit: at default `verify_n`, top-1 matches dense for canonical
      prompts.
    - Integration: tied + untied paths both work.
    - Integration: each sampler (greedy / top-K / top-P / repetition
      penalty) produces sensible output.
    - Regression: mock MTP test confirms no shape mismatch with the
      auxiliary head's expected `lm_head` interface.
22. **Documentation.**
    - User-facing: README entry on what the feature does, when to
      enable, expected speedup, configuration knobs.
    - Caveats: repetition-penalty interaction; tied-embedding support;
      MTP behaviour.
    - Method comparison: PCA vs RP, when to use each.
23. **Benchmark script.** `examples/bench_approx_lm_head.py` reproducing
    the §5.18 measurement.
24. **PR.** Title: *"feat: ApproxLMHead — large-vocab lm_head
    acceleration for Qwen 3.5 family (and other large-vocab models)"*.
    Body should include the discussion-issue link, recall + quality
    + speedup measurements, opt-in mechanism, list of variants
    tested, citation of the LARQL math docs.
25. **Iterate on review.** Expect questions about:
    - PCA vs RP (have a per-method recall comparison ready).
    - Tied embeddings (have the integration test for both cases).
    - MTP interaction (have the §5.15 measurement).
    - Memory determinism across mlx versions (PCA is deterministic
      from the weight; RP is deterministic from the seed).
    - Worst-case recall (have p95/min/p1 numbers, not just means).

---

## 6. Validation criteria (definition of done)

| Criterion | Target | How measured |
|---|---|---|
| Top-50 jaccard recall (mean) | ≥ 0.99 | Recall sweep on 1,000 prompts |
| Top-50 jaccard recall (p95) | ≥ 0.97 | Same |
| **Top-10 jaccard recall (mean)** | **≥ 0.95** | **Same** |
| **Mean KL divergence (top-P=0.95 mass)** | **≤ 0.01 nats** | **Same** |
| MMLU accuracy drift | ≤ 0.5% absolute | 1,000-q subset |
| HellaSwag accuracy drift | ≤ 0.5% absolute | 500-q subset |
| Greedy parity rate | ≥ 99% | 50 prompts × 200 tokens |
| **MTP accept-rate drift** | **±5% of dense** | **50 prompts × 100 tokens** |
| Total decode speedup (122B-A10B) | **≥ 5%** (target), 8% (stretch) | 20-trial median, 200-tok greedy |
| Load-time projection cost (122B) | ≤ 60 s (PCA) | Wall clock |
| **Memory overhead (absolute)** | **≤ 50 MB** | **RSS before/after** |
| Tied-embedding integration | Works on Qwen 3.5-9B | Smoke test + recall sweep |
| Untied-embedding integration | Works on Qwen 3.5-122B-A10B | Same |
| Tests pass | 100% | CI |

If any criterion is not met after tuning, document the gap honestly in
the PR. Bold rows are *new or tightened* relative to the original plan.

---

## 7. Reference materials

### 7.1 Internal docs (LARQL repository, corrected paths)

- [`docs/synthesis/01-explainer.md`](01-explainer.md) — LARQL conceptual overview.
- [`docs/synthesis/02-mathematical-foundation.md`](02-mathematical-foundation.md)
  — sections 2–4 ground the math. The walk-FFN derivation in §2 is the
  exact analogue of what we are doing here for `lm_head`.
- [`docs/synthesis/03-software-architecture.md`](03-software-architecture.md) —
  how LARQL organises gate-KNN dispatch.
- [`docs/synthesis/05-questions.md`](05-questions.md) — Q1 §"Where the cost
  saving actually appears" is directly relevant.
- [`docs/walk-boundary-sweep.md`](../walk-boundary-sweep.md) — empirical
  proof that approximate-then-verify preserves output (Gemma 3-4B,
  zero divergence at all 34 layer boundaries).
- [`docs/ffn-graph-layer.md`](../ffn-graph-layer.md) — engineering
  details of LARQL's analogous primitive.

### 7.2 External references

- **Halko, Martinsson, Tropp 2011.** *"Finding Structure with Randomness:
  Probabilistic Algorithms for Constructing Approximate Matrix
  Decompositions."* The randomised SVD algorithm used in
  `_build_pca_projection`.
- **Johnson-Lindenstrauss 1984; Achlioptas 2003** — JL random projection.
  Used as the RP fallback.
- **Indyk & Motwani 1998** — approximate nearest neighbours.
- **MLX docs.** <https://ml-explore.github.io/mlx/build/html/index.html>.
- **MLX-LM repo.** <https://github.com/ml-explore/mlx-lm>.
- **Qwen3-Next paper / blog.** <https://qwen3-next.com/>.

### 7.3 Code touchpoints in MLX-LM

To be confirmed at sprint start:

- `mlx_lm/models/qwen3_5.py` (or canonical Qwen 3.5 model file).
- `mlx_lm/utils/approx_lm_head.py` (new file).
- `mlx_lm/utils/__init__.py` — export `enable_approx_lm_head`.
- `mlx_lm/sample_utils.py` / `mlx_lm/generate.py` — verify samplers
  tolerate `-inf` logits.
- `tests/test_approx_lm_head.py` (new).
- `examples/bench_approx_lm_head.py` (new).

---

## 8. Pitfalls and watch-outs

(Substantially extended from the original plan.)

### 8.1 MLX lazy evaluation

The `weight @ self.proj_matrix.T` projection MUST be eagerly evaluated
at construction time via `mx.eval(...)`. If MLX defers it, every forward
pass rebuilds the projection and the optimisation is worse than dense.
The implementation in §4.1 has explicit `mx.eval` calls; do not remove
them.

### 8.2 Tied vs untied embeddings

**Already addressed** in §4.2. Don't defer this; Qwen 3.5-9B (the dev
target) is likely tied. Validate the tied path before benchmarking.

### 8.3 Quantised weights

If the model is loaded with quantisation (Q4 / Q8 / NVFP4), `model.lm_head.weight`
may not be a plain f16/f32 array. Two choices:

- **De-quantise once** to f16 inside `ApproxLMHead.__init__`. Adds
  memory (one f16 copy of `lm_head` ~= 2.5 GB on 122B), but simplifies
  the verify-stage gather. **Acceptable on 122B+ where the model is
  already large; not acceptable on 9B Q4 where it would exceed the
  full-precision lm_head size.**
- **Stay quantised**: use whichever dequant kernel `nn.QuantizedLinear`
  uses for the verify-stage gather. More work, no extra memory. **Recommended.**

For the initial PR, ship the second option for both tied and untied
quantised cases.

### 8.4 Stage 2 random-row gather efficiency on Metal

Random-row gather of 2,048 rows from a 248,320-row matrix doesn't hit
peak bandwidth (see review §2). Consider:

- Use `mx.take(weight, cand, axis=0)` rather than fancy indexing —
  `take` has a more predictable Metal kernel path.
- If Stage 2 becomes the bottleneck, sort `cand` before gather; this
  improves spatial locality (touching nearby rows together is faster
  than touching arbitrary rows). Effect is modest but free.
- A custom Metal kernel that fuses gather+matmul into one pass would
  recover most of the lost bandwidth. **Defer to follow-up** — the
  initial PR ships unfused.

### 8.5 First-token vs subsequent-token cost

Prefill batches many tokens through `lm_head` at once and benefits
proportionally less. The optimisation primarily helps decode. Make
sure benchmarks measure decode tokens-per-second after the first
output token. MLX-LM's existing benchmark scaffold typically does
this correctly; verify.

### 8.6 Numerical stability with -inf

`-inf * 0 = NaN`. Confirm no sampler does `mask * logits` where
`mask` could be 0 at a `-inf` position. Standard MLX-LM samplers use
`logits + mask` where `mask` is `0.0` or `-inf` — no `NaN` risk.

The `_inf_buffer` is f32; even when `x` is f16, the output is upcast
to f32 to avoid `f16(-inf) → -65504` issues with subsequent softmax.

### 8.7 Don't break existing models

All changes are additive. If `enable_approx_lm_head` is never called,
behavior is byte-identical to before. Maintain ruthlessly.

### 8.8 Multi-Token Prediction interaction (NEW)

The MTP head is unchanged but its dynamic with approximate main-`lm_head`
matters. See §5.15 and §4.5. Measure accept rate; report in PR.

### 8.9 Don't redesign without authorization

The user has authorised PCA *and* RP. Do not silently switch to HNSW,
FAISS, product quantisation, or any other algorithm without surfacing
and asking. The `target_dim` and `verify_n` knobs are yours to tune
within the documented ranges; the algorithm choice is yours within
{"pca", "rp"}.

### 8.10 Benchmark hygiene (NEW)

- Median of 20 trials, IQR reported (not 5 trials with a single number).
- `caffeinate -i` to prevent sleep-state changes.
- 30-second warm-up before first timed run.
- Disable Spotlight, Time Machine, background apps.
- Pin to P-cores via `taskpolicy` if available.
- Report tok/s, time-to-first-token, and decode-only tok/s separately.

---

## 9. Communication and coordination

- **The user (project owner):** has approved this plan. Provides
  hardware access (M3 Ultra Mac Studio for 122B benchmarks) and HF
  credentials.
- **MLX-LM maintainers:** **a discussion issue is required** before
  the PR (Phase 1, step 5). The change touches a new module, a public
  helper, sampler-relevant output format, and (for tied embeddings) a
  forward-path patch — large enough to warrant pre-coordination.
- **You (the agent):** end-of-phase status updates to the user.
  Surface immediately on:
  - Tied-embedding path doesn't work on Qwen 3.5-9B.
  - PCA build cost > 60 s on 122B (decide RP fallback or larger
    constraint).
  - MTP interaction degrades accept rate by > 10%.
  - Any of the §6 quality criteria fail after tuning.
- **LARQL repository owner:** Chris Hayuk. The user coordinates
  cross-references back to this repo.

---

## 10. Beyond this sprint (out of scope for v1)

In priority order, after the v1 PR lands:

1. **Custom Metal gather-fused-matmul kernel** for Stage 2. Recovers
   the ~30% of theoretical speedup lost to gather inefficiency. Most
   impactful follow-up.
2. **Fully sparse output mode.** Return `(cand, exact_scores)` and patch
   samplers to consume sparse logits directly. Eliminates the
   per-token full-V buffer write.
3. **Apply to Qwen 3.5-397B-A17B** (deferred from v1 sprint).
4. **Apply to gate-KNN per-layer FFN selection.** LARQL-style at the
   FFN level, larger total wins on top of the lm_head speedup.
5. **Sidecar mode for production deployment.** Once the technique is
   proven in production, allow saving the projection to disk for
   instant load.
6. **MTP head extension.** Apply ApproxLMHead to Qwen 3.5's MTP head
   itself.
7. **Validate on Kimi K2 / GLM-4.5 / DeepSeek V3.** Same algorithm,
   different model families. Expect similar speedups.
8. **Port to llama.cpp.** Larger user base, harder PR. Effectively a
   fresh implementation, not a port — set expectations accordingly.

None of the above is in scope for v1. Ship the simple version cleanly.

---

## 11. One-line summary

> Replace Qwen 3.5's `lm_head` projection in `mlx-lm` with a two-stage
> approximate-then-exact search: a PCA-based 64-dim prefilter (random
> projection as fallback) over `[V × H]` followed by exact dot products
> on the top-2,048 candidates. Built at load time in 10–30 s, no
> sidecar files. Mathematically grounded in randomised SVD and JL;
> engineering-grounded in LARQL's gate-KNN pattern. Realistic target:
> ~5% total decode speedup on Qwen 3.5-122B-A10B (8% stretch),
> negligible quality loss, no surprises with tied embeddings or MTP.

The math is sound, the engineering is bounded, the quality safeguards
are in place. Ship cleanly.
