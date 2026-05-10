# Review: MLX LM Head Acceleration via Random Projection

> Review of the MLX-LM `ApproxLMHead` sprint plan. The core idea is sound and
> the plan is unusually well-grounded for a one-shot AI proposal, but several
> technical details are glossed over in a way that could cause the sprint to
> miss its targets if executed verbatim. This document enumerates the issues;
> a revised plan is in [`07-mlx-plan-revised.md`](07-mlx-plan-revised.md).

## Overall assessment

**The technique is correct, the targeting is correct, and the overall sprint
plan structure is good.** The plan correctly identifies that:

- Large-vocab `lm_head` is a real bandwidth bottleneck on Mac.
- Sampling needs only the top of the distribution, not the bottom 99%.
- Two-stage approximate-then-exact recovers exact results when verify_n is
  large enough.
- Qwen 3.5's hybrid-attention + ultra-sparse-MoE makes `lm_head` a
  proportionally larger share of decode time than on dense models.

These are the load-bearing claims, and they hold up.

**Where the plan needs work** is in three areas:

1. **Engineering realism** — the bandwidth math glosses over scatter/gather
   inefficiency, kernel launch overhead, and the reality that "tall skinny"
   matmuls on Metal don't hit peak FLOPs. The headline "47× bandwidth
   reduction → 8–12% speedup" is plausible but optimistic; 4–8% is a more
   realistic baseline expectation.
2. **Algorithm choice** — random projection is presented as the solution,
   but PCA (top-k right singular vectors) gives ~2–4× better recall at the
   same `target_dim` for the same one-time build cost. The plan mentions
   PCA only as a "fallback if RP underperforms"; it should be the primary.
3. **Edge-case handling** — tied embeddings (the *common* case for the dev
   target Qwen 3.5-9B), Multi-Token Prediction head interaction, and
   speculative-decoding accept-rate are all deferred or undersold. At
   least tied embeddings and MTP need to be first-class concerns from
   day one for Qwen 3.5 specifically.

A successful sprint executing the plan-as-written has a high chance of
shipping *something* that works but a moderate chance of missing the
8% speedup target on the 122B variant. With the revisions in
[`07-mlx-plan-revised.md`](07-mlx-plan-revised.md), the chance of hitting
the target rises substantially.

## What the plan gets right

A short list, because the strong points genuinely are strong:

- **Math grounding.** The Johnson-Lindenstrauss treatment in §2.4 is
  accurate and appropriately calibrated about the difference between
  worst-case bounds and observed empirical performance.
- **Connection to LARQL's gate-KNN pattern.** The analogy is exact and
  the pattern's success on FFN gates is real validation evidence for
  the proposal. Citing the boundary sweep is the right move.
- **Phased validation plan.** Develop on 9B for fast iteration, validate
  on 122B for the headline number is the correct sequencing.
- **Opt-in by default.** Wrapping after model construction (Approach A)
  is the right shape for a first PR. Easier to review, easier to roll
  back, doesn't perturb existing model loads.
- **Sparse-logit interface compatibility.** Returning `[..., V]` with
  `-inf` outside the candidates is the right choice for sampler
  compatibility, even if it means materialising a full-V output.
- **The pitfalls section.** The `mx.eval()` warning in §8.1 is exactly
  the kind of thing a fresh agent will get bitten by if not flagged.

## Technical issues, in order of severity

### 1. Tied embeddings should not be deferred

The plan says (§4.2):
> *"Tied embedding case: this is more invasive; defer to Approach B"*

This is wrong for the actual development target. Qwen 3.5-9B almost
certainly uses tied embeddings (most ≤30B open-weight models do, including
Qwen 3.0-7B and likely the smaller Qwen 3.5 variants). The plan therefore
*can't be smoke-tested* on the development target with the proposed
Approach A.

**Fix:** Handle tied embeddings as a first-class case in the initial
implementation. The `ApproxLMHead` module just needs to accept a
weight reference; whether that weight is owned by `model.lm_head`
or by `model.embed_tokens` doesn't change the algorithm. The
integration helper should detect tied vs untied and wrap accordingly:

```python
def enable_approx_lm_head(model, ...):
    if model.lm_head is not None:
        # Untied: standard wrap
        weight = model.lm_head.weight
        model.lm_head = ApproxLMHead(weight, ...)
    else:
        # Tied: weight is in embed_tokens; wrap the call site
        weight = model.embed_tokens.weight
        approx = ApproxLMHead(weight, ...)
        # Monkey-patch model.__call__ or install a forward hook; specifics
        # depend on MLX-LM's model class structure
        model._approx_lm_head = approx
        # ... wire it into the forward path
```

This is not "more invasive" by any meaningful measure — it's one extra
branch in the loader.

### 2. The bandwidth analysis omits gather inefficiency

The plan's cost table (§2.5) treats Stage 2's "exact verify" as a clean
21 MB read:
> *"Stage 2 (exact verify) | 2048 × 5120 ≈ 10.5 M ops | ~21 MB read"*

In practice this is a **gather of 2048 random rows from a 248,320-row
matrix**. Each row is 10 KB. Random-row gathers on Metal:

- Don't coalesce well (each row starts at a different cache line).
- Defeat hardware prefetching (no spatial locality across rows).
- Don't approach peak memory bandwidth on real hardware.

A reasonable rule of thumb: gather throughput on Apple Silicon is
~30–50% of contiguous-read bandwidth for randomly-distributed indices.
So the "21 MB" is closer to "21 MB at ~150 GB/s effective" instead of
"21 MB at 400 GB/s effective." That makes Stage 2 cost ~140 µs instead
of ~50 µs.

For Stage 1, the V × d matmul has two sub-issues:
- `[V, d]` with d=64 is "tall skinny" — kernels optimised for square
  matmuls underperform on this shape, often by 2–4×.
- The argpartition over a 248K-element array is a separate kernel
  that contributes its own latency (~100–300 µs on Metal for a top-N
  partial sort).

**Net realistic cost** for both stages combined: ~400–700 µs vs the
~3–6 ms dense baseline. That's still a 5–10× improvement over dense
on lm_head specifically — meaningful, but not the headline 47×.
Translating to total decode speedup: realistic range is **4–8%**, not
8–12%.

**Fix:** The plan should set the speedup target at 5% and treat 8%+ as
a stretch goal, *not* the definition of done.

### 3. PCA should be the primary, not the fallback

Random projection's worst-case bound (the JL calculation in §2.4) gives
`d ≈ 1700` for `ε = 0.1, δ = 0.001` — but the plan correctly notes that
empirically `d = 64` works because the bound is loose.

PCA does even better. The right singular vectors of `lm_head` capture
the actual subspace `lm_head` lives in — typically the top ~50–200
singular values contain >95% of the energy, even on huge embedding-style
matrices. Empirically:

- Random projection at d=64: top-50 jaccard ≈ 0.95–0.99
- PCA at d=64: top-50 jaccard ≈ 0.99–0.999

For a similar one-time build cost (truncated randomized SVD, ~10–30 s
on a 248K × 5120 matrix on Metal), PCA gets you the same recall with
much smaller `verify_n`, or much higher recall with the same
`verify_n`. That directly translates to either smaller Stage 2 cost
(more speed) or smaller deviation from dense (more quality safety
margin).

**Fix:** Make PCA the primary algorithm. Random projection becomes the
fallback if SVD is too expensive on a particular model size, or if
the user explicitly wants reproducibility from a seed without reading
the weights at all.

### 4. Multi-Token Prediction head interaction is undersold

The plan (§4.4) says:
> *"Don't touch MTP. Verify MTP still works in the integration test."*

This understates the interaction. Qwen 3.5's MTP is used for
*self-speculative decoding*: the model produces multiple candidate
tokens per forward pass via the auxiliary heads, then verifies them
using the main lm_head. The verification step asks: *is the main
lm_head's distribution at this position consistent with the drafted
token?*

If the main lm_head is now approximate, the verification distribution
is also approximate, which:
- **Changes the speculative-decoding accept rate.** If approximation
  makes the verification distribution slightly less peaked, more drafts
  reject; speed goes down.
- **May change generated outputs** even at temperature 0, because
  drafting at one set of logits and verifying at a different set is
  not the same as the original system.

This isn't necessarily a *blocker* — the change might be small or
even neutral — but it must be measured. The plan should add a
phase-3 step:
- Run the model with MTP enabled and disabled, both dense and approx.
- Compare: tokens/second, output identity at temp=0, accept rate.
- If accept rate drops by >5%, surface and discuss with the user.

### 5. "Greedy parity ≥ 99%" is too easy a target

For typical prompts, the top-1 logit dominates by orders of magnitude
in pre-softmax space. Random projection preserves relative rankings
well enough at the top that 99% greedy parity is essentially given.

The harder, more meaningful test is **distribution similarity** under
sampling. Specifically:

- For each of N prompts at temperature 1.0, compare the post-softmax
  distributions (restricted to the top-P=0.95 cumulative mass) between
  dense and approx.
- Report KL(approx || dense) and KL(dense || approx). Both should be
  small (< 0.01 nats is a reasonable bar).
- Also report top-K agreement at K=10 (much harder than K=50; tests
  the head of the distribution where sampling actually picks).

**Fix:** Add the KL-divergence check to validation. Keep the greedy
parity check but downgrade it from "definition of done" to "sanity
check."

### 6. "≤ 5% memory overhead" is way too lax

The plan's memory budget allows 5% of model RAM as projection
overhead. On a 244 GB f16 122B model, that's 12 GB. The actual
projection memory is:

- `proj_matrix` at d=64: 64 × 5120 × 4 bytes = 1.3 MB
- `weight_proj` at d=64, V=248K: 248K × 64 × 2 bytes = 32 MB
- Total: ~33 MB

That's 0.013% of the model, not 5%. A 5% budget is permissive enough
to accommodate "we accidentally kept a copy of the full weight in
memory," which would be a bug, not a budget.

**Fix:** Set the memory budget at 0.1% of model RAM, or absolute
50 MB, whichever is larger. Anything more is a defect.

### 7. The sparse `[..., V]` output isn't free

The plan's `__call__` returns `mx.full(approx_scores.shape, -mx.inf, ...)`
then scatters exact_scores into it. That's a per-token allocation of
a full V-sized tensor (~500 KB at f16, 248K elements).

This isn't a *correctness* issue, but it has implications:
- Allocation + initialise + scatter of 500 KB per token adds up at
  high tok/s.
- The whole point of the optimisation is to *not* touch a full-V
  buffer per token; the output materialisation undoes some of that.

**Fix:** Two options. (a) Keep the full `-inf` output for sampler
compatibility, but pre-allocate it as a buffer and reuse across
tokens. (b) Add an opt-in "sparse-output mode" that returns
`(candidate_idx, candidate_logits)` and patch the samplers to handle
the sparse form. Option (b) is more invasive but recovers the last
~10–20% of the achievable speedup.

For the initial PR, ship (a) with the buffer reuse. Note (b) as a
follow-up.

### 8. Benchmark methodology is too lenient

Five trials with median is too few for inference benchmarks on Mac,
which has high run-to-run variance from:
- Thermal throttling (especially after warm-up)
- Background processes (Spotlight indexing, Time Machine)
- E-core vs P-core scheduling decisions

**Fix:**
- Median of 20 trials minimum, reported with IQR.
- Disable Spotlight, Time Machine, etc. during benchmarking.
- Use `caffeinate -i` to prevent power-state changes.
- Run an explicit warm-up of 30 seconds before timing.
- Pin to P-cores via `taskpolicy` if available.

This is standard inference-benchmarking hygiene; the plan should
codify it.

## Process and methodology concerns

### A. "Don't redesign without authorization" is reasonable but...

The plan in §8.8 says:
> *"The user has approved the random-projection approach specifically.
> If you discover a fundamental blocker, surface it and ask. Do not
> silently switch to HNSW, FAISS, or another algorithm."*

The instruction is reasonable, but the threshold for "ask" is
unclear. Switching to PCA (still a Gaussian-style projection — just
the *informed* one) is sufficiently close to RP that an agent could
reasonably argue either way. The plan should explicitly authorise
PCA as an alternative, or explicitly forbid it.

I'd recommend: **explicitly authorise PCA**, since it's strictly
better-conditioned for this problem.

### B. MLX-LM upstream coordination is undersold

§9 says "open a discussion issue *before* the PR if the change
touches more than the model file + a new utility." This change does
touch more than that — it adds a new module, a public helper
function, a documentation entry, and (likely) modifies the sampler
or generate path. **A discussion issue is not optional**; it's the
right way to introduce a feature of this size to the upstream
project.

### C. The dev environment lacks a quality regression gate

The plan validates against MMLU/HellaSwag *once* at the end. It
should validate at multiple settings during development to catch
regressions early. Even running 100-question subsets at the end of
each phase is enough to catch "oh, we broke something at d=64"
before it's compounded with five other changes.

### D. The 397B variant should be the primary benchmark target, not a stretch

§5.15 lists Qwen 3.5-397B-A17B as an optional stretch. This is
backwards for the actual user's situation: the user's M3 Ultra Mac
Studio has 512 GB unified memory and **397B is one of the models
they routinely run locally**. It is precisely the workload the
optimization is meant to accelerate. The 122B-A10B variant should
be the smaller, faster-iteration *secondary* target — useful for
pinning down knobs cheaply — but the headline speedup measurement
should be on 397B-A17B.

There's also a stronger architectural argument: at 397B with top-K
MoE active params of ~17B, FFN bandwidth is even more dominated by
expert dispatch than on 122B. The relative share consumed by
`lm_head` is *higher*, so the optimization's payoff is bigger on
397B than on 122B. The 5–8% total-decode speedup target is most
defensible on the 397B variant.

The revised plan promotes 397B to primary and demotes 122B to
"intermediate validation" rather than deleting it.

## Section-specific minor comments

| Section | Issue |
|---|---|
| §2.1 | "Mac Studio M3 Ultra ~200 GB/s effective CPU memory bandwidth" — actually closer to 800 GB/s on Ultra. The plan is using a Pro-tier number. |
| §2.4 | The JL bound `d = 8 log(2/δ)/ε²` is for *one* inner product. The union bound over V inner products multiplies the failure probability; the plan does this correctly but the exposition is a little compressed. |
| §2.5 | Cost table omits the per-token query projection `q @ P.T` — negligible at H × d ≈ 0.3 M ops, but should be in the table for completeness. |
| §4.1 | `(weight @ self.proj_matrix.T).astype(mx.float16)` will materialize lazily; the comment says "// mx.eval... if eager evaluation matters" — it *does* matter, drop the conditional and always `mx.eval`. |
| §4.1 | `mx.full(approx_scores.shape, -mx.inf, dtype=x.dtype)` should use `mx.float32` regardless of `x.dtype` to avoid `f16(-inf) → -65504` issues with later softmax. |
| §4.4 | The sentence "MTP is a smaller secondary head" is misleading — MTP does affect the main decode pipeline because it gates speculative acceptance. |
| §5.4 | The baseline benchmark should also report variance, not just a single tok/s number. |
| §5.9 | "Mix of chat / code / multilingual" — should explicitly include long-context prompts since lm_head latency is per-token regardless of context length. |
| §5.12 | "First, increase target_dim from 64 → 128" — should also try 32 first (often works fine on embedding matrices and halves Stage 1 cost). |
| §6 | "Memory overhead ≤ 5% of model RAM" — see issue #6 above; tighten to 0.1% / 50 MB. |
| §10.1 | "Port to llama.cpp" is named as #1 follow-up but llama.cpp's BPE/sampling stack is sufficiently different that it's effectively a fresh implementation, not a port. Set expectations accordingly. |

## Recommendation

**Don't execute the plan as written.** The core idea is right; the
plan has enough engineering optimism baked in that an agent
following it literally is likely to over-promise the speedup and
under-handle two edge cases (tied embeddings, MTP). Both are
fixable.

**Do execute the revised plan in
[`07-mlx-plan-revised.md`](07-mlx-plan-revised.md),** which:

- Promotes PCA to primary algorithm (RP as fallback).
- Handles tied embeddings from day one.
- Adds an MTP-interaction phase to Phase 3.
- Promotes Qwen 3.5-397B-A17B to the primary perf target (matches
  the user's actual workload on a 512 GB M3 Ultra Mac Studio);
  keeps 122B as a faster-iterating intermediate target.
- Tightens benchmark methodology and validation criteria.
- Resets the speedup target to 5%, with 8% as a stretch.
- Tightens the memory budget.

The revised plan is structurally the same — same sprint duration
(~2 weeks), same target model, same definition of done — just with
the load-bearing engineering details made explicit.

## A note on the broader ambition

The plan correctly observes that Qwen 3.5's hybrid-attention +
ultra-sparse-MoE architecture makes `lm_head` a relatively bigger
share of decode time. This is part of a general trend: as models
get more sparse in the *middle* (MoE, linear attention, walk-FFN),
the *ends* (embeddings, lm_head, normalisation) become proportionally
more expensive.

That makes this kind of optimisation more valuable on every new
generation of architecturally-novel models, not less. After this
PR ships, the same approach should apply to:

- DeepSeek V3/V4 (vocab=128K, similar sparsity profile)
- Kimi K2.6 (vocab~163K)
- GLM-4.5/4.6 (vocab~155K)

So the sprint isn't just a Qwen-specific optimization — it's a
prototype for a category of work. That deserves a paragraph in the
final PR description, because it makes the optimization much more
attractive to the MLX-LM maintainers (one new utility that helps
six model families is a much easier merge than one Qwen-specific
hack).
