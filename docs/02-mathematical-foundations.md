# LARQL — Mathematical Foundations

> A self-contained derivation of the FFN-as-graph identity, the walk-FFN equivalence, the constellation insertion method, and the COMPILE write-back. Intended as a foundation upon which new implementations can be built.

---

## 1. Notation and assumptions

We work in `ℝ^h` (residual space) and `ℝ^m` (intermediate / FFN space), with `h` the hidden size and `m` the intermediate size. Vectors are columns by default; `xᵀ y` is the inner product, `x ⊙ y` the elementwise (Hadamard) product, `‖x‖` the L2 norm, and `x̂ = x / ‖x‖` the unit normalization.

A transformer block consumes a residual `x ∈ ℝ^h` and produces `x' = x + δ_attn + δ_ffn`. We focus exclusively on `δ_ffn`; attention is treated as an opaque routing operator that produces the input residual we feed into the FFN.

**Gated FFN** (the form used by Gemma 2/3/4, Llama 2/3, Mistral, Qwen, Phi, Mixtral experts, GeGLU/SwiGLU):

```
δ_ffn(x) = W_down · ( σ(W_gate · x) ⊙ (W_up · x) )                                        (1)
```

with weights `W_gate, W_up ∈ ℝ^{m × h}`, `W_down ∈ ℝ^{h × m}`, and `σ` an elementwise nonlinearity (SiLU for SwiGLU, GELU-tanh for GeGLU).

We will index features by `i ∈ {0, …, m-1}`. We define:

- **Gate vector** `g_i ∈ ℝ^h` — the `i`-th row of `W_gate` (transposed to a column).
- **Up vector** `u_i ∈ ℝ^h` — the `i`-th row of `W_up`.
- **Down vector** `d_i ∈ ℝ^h` — the `i`-th column of `W_down`.

Each FFN feature `i` is the triple `(g_i, u_i, d_i)`.

**Dense FFN models** (GPT-2, original Transformer):

```
δ_ffn(x) = W_out · σ(W_in · x)                                                            (2)
```

Here there is no gate / up split; we treat `W_in` rows as gate vectors and `W_out` columns as down vectors with `u_i = g_i` (effectively). The graph identity below applies with one minor specialization.

---

## 2. The matmul-as-graph identity

Substituting the per-feature decomposition into (1):

```
δ_ffn(x) = W_down · ( σ(W_gate · x) ⊙ (W_up · x) )
        = ∑_{i=0}^{m-1} d_i · σ(g_iᵀ x) · (u_iᵀ x)                                        (3)
        = ∑_{i=0}^{m-1} a_i(x) · d_i
```

where the per-feature **activation**

```
a_i(x) = σ(g_iᵀ x) · (u_iᵀ x)                                                             (4)
```

is a scalar function of the residual `x` only.

**Reading (3) as a graph operator.** Define a directed labeled hypergraph `G_L = (V, E)` for layer `L`:

- Vertices `V` = points of the unit sphere `S^{h-1}` ⊂ `ℝ^h` (the directional content of residuals).
- Edges `E` = `{ e_i = (g_i, d_i, a_i(·)) : i = 0..m-1 }`. Edge `e_i` reads from direction `g_i` (via the inner product), writes to direction `d_i` (additively), with weight `a_i(x)`.

Then the FFN is exactly:

```
δ_ffn(x) = ∑_{e_i ∈ E} a_i(x) · d_i                                                       (5)
```

This is a linear superposition of edge writes, where each edge's write strength is a function of the input. **The matrix multiplication is the sum over the graph.** No approximation, no quantization, no information loss — this is an algebraic restatement of (1).

The whole FFN stack across the network is the composition `δ^{(L-1)} ∘ … ∘ δ^{(1)} ∘ δ^{(0)}` plus residual additions and attention deltas. **The full transformer is the iterated graph operator.**

---

## 3. Sparsity follows from the activation

The activation `a_i(x)` is sparse-by-design in trained transformers because:

- **SiLU saturates negative gates.** For `g_iᵀ x < 0` and large in magnitude, `σ(g_iᵀ x) ≈ 0`, so `a_i ≈ 0`.
- **Up-gate alignment.** Trained features have `u_i ≈ g_i` direction-wise, so a feature that doesn't recognize `x` (small `g_iᵀ x`) also has small `u_iᵀ x`, multiplying the suppression.
- **Empirical sparsity.** At any given layer, at most ~10–20% of features have `|a_i(x)| > τ` for any practical threshold `τ`. (Measured for Gemma 3 4B at `top_k=8092` of `m=10240` features captures essentially the entire informative tail.)

Define the **active set** for input `x`:

```
A_K(x) = { top-K indices of |g_iᵀ x| over i = 0..m-1 }                                    (6)
```

(We rank by `|g_iᵀ x|` rather than `|a_i(x)|` because `g_iᵀ x` is computable in one BLAS gemv before evaluating any `u_i` or `σ`. The rank correlation with `|a_i|` is empirically near-perfect once features with same-direction `u_i` are common.)

The **walk-FFN** approximates (3) by

```
δ̃_ffn(x) = ∑_{i ∈ A_K(x)} a_i(x) · d_i                                                   (7)
```

For `K = m`, (7) equals (3) exactly — no terms are dropped. For `K < m`, (7) drops the lowest-magnitude `m-K` terms; the resulting error is bounded by the L2 norm of the dropped activations:

```
‖δ_ffn(x) - δ̃_ffn(x)‖ ≤ ‖∑_{i ∉ A_K(x)} a_i(x) · d_i‖
                       ≤ √( ∑_{i ∉ A_K(x)} |a_i(x)|² · ‖d_i‖² )                            (8)
```

In practice, with `K = 0.8 · m` on Gemma 3 4B, the per-layer L2 error is ~10⁻³ relative to ‖δ_ffn‖, and **top-1 token agreement after softmax is exactly 100% on a 5-prompt sweep at every layer-boundary mix** (see the boundary sweep in `walk-boundary-sweep.md`).

---

## 4. Gate KNN as the graph index

Equation (6) defines the active set as the top-K of an inner product. This is precisely a **maximum inner product search** (MIPS) problem on the rows of `W_gate`. In LARQL:

```
gate_knn(L, x, K):
    G_L ∈ ℝ^{m × h}        # mmap'd, layer L's W_gate
    s = G_L · x            # one BLAS gemv: m dot products at once
    return argmax-K |s|
```

This is one BLAS `cblas_sgemv` call per layer. On Gemma 3 4B (`m=10240, h=2560`) it takes ~0.008 ms on M-series CPU. For a 34-layer walk, ~0.3 ms total. Optional acceleration:

- **HNSW index** on `g_i / ‖g_i‖` for `m > ~50k` (becomes worthwhile when MoE puts experts × features into one layer).
- **Q4_K dequantize-and-multiply** kernels for low-precision storage.
- **Metal GPU** dispatch for batched MIPS over multiple layers.

The KNN view also gives a clean semantics for **feature labels**: a feature `(g_i, d_i)` is "the edge that fires on direction `g_i` and writes direction `d_i`". The label is found by

- forward-direction probing: `top-k` tokens by `embed(t)ᵀ g_i / ‖embed(t)‖` (what activates this feature?);
- backward-direction probing: `top-k` tokens by `embed(t)ᵀ d_i / ‖embed(t)‖` (what does it produce?).

The down-side probe is the basis of `down_meta.bin`: per feature, store the top-K vocabulary tokens with logit scores. That metadata is what `DESCRIBE` reads.

---

## 5. Circuit-type classification

A simple cosine measurement on `(g_i, d_i)` partitions the graph's edges into mechanistic types:

```
ρ_i = cos(g_i, d_i) = (g_iᵀ d_i) / (‖g_i‖ · ‖d_i‖)                                        (9)
```

| Range | Type | Behaviour |
|-------|------|-----------|
| `ρ_i > 0.5` | Identity | Reads `x` along `g_i`, writes back along `d_i ≈ g_i`. Self-reinforcement. |
| `0.2 < ρ_i ≤ 0.5` | Transform | Related forms (morphological, syntactic). |
| `-0.2 ≤ ρ_i ≤ 0.2` | Projector | Orthogonal in/out. **Factual bridges.** Most features. |
| `-0.5 ≤ ρ_i < -0.2` | Suppressor | Weak direction flip (gating, interference). |
| `ρ_i < -0.5` | Inverter | Strong direction flip (format enforcement). |

Empirically on Gemma 3 4B, the projector fraction tracks the "knowledge layers" (L19–L29 peak at 85–95% projector); identity+inverter fraction tracks the "format gate" (L30–L33 peak at 11% combined). This is computed in `O(m · h)` per layer, no forward pass needed.

The classification is purely a property of the trained weights, but it gives an immediate semantic decomposition: **factual edges live in the projector subset of the graph.**

---

## 6. The constellation insertion method (training-free fact insertion)

We now derive the procedure used by `INSERT INTO EDGES`. The goal is: given a vindex with weights `(W_gate, W_up, W_down)` and a target fact `(entity, relation, target)` such as `(Atlantis, capital-of, Poseidon)`, modify a small set of FFN features so that the model produces `target` on the canonical prompt `"The {relation} of {entity} is"` — without retraining and without breaking neighbouring facts.

### 6.1 The constraint

Let `T(x) = "The capital of Atlantis is"` produce the residual `r_L ∈ ℝ^h` after attention at layer `L`. Let `e_T = embed("Poseidon")` be the answer's token embedding. With **tied embeddings** (Gemma, Llama), the logit for token `t` after the final norm is `lm_head_t · final_residual = embed(t)ᵀ · final_residual / logits_scale`.

We want the cumulative FFN contributions from layer `L_0` to `L_{end}` to push the residual along `e_T`:

```
∑_{L ∈ install_layers} δ_ffn^{(L)}(r_L) ≈ β · ê_T                                          (10)
```

for some `β > 0` large enough to make `e_T`'s logit dominate. Equivalently, we want each chosen layer's FFN to contribute `α · ê_T` (rescaled to layer-typical magnitude), with `∑ α = β`.

### 6.2 One feature per layer

At each install layer `L`, we select an unused slot `i*` and write:

```
g_{i*}^{(L)} = r̂_L · g_norm^{(L)} · S                                                    (11a)
u_{i*}^{(L)} = r̂_L · u_norm^{(L)}                                                        (11b)
d_{i*}^{(L)} = ê_T · d_norm^{(L)} · α                                                    (11c)
```

where `g_norm^{(L)}, u_norm^{(L)}, d_norm^{(L)}` are the median row/column norms at this layer (preserves "natural" magnitude regime), `S = 30` (the empirically validated gate scale), and `α ≈ 0.25` per layer.

The activation at slot `i*` for input `x` is

```
a_{i*}(x) = σ((g_{i*})ᵀ x) · ((u_{i*})ᵀ x)
         = σ(g_norm · S · r̂_Lᵀ x) · (u_norm · r̂_Lᵀ x)                                  (12)
         ≈ σ(S · g_norm · cos(r_L, x) · ‖x‖) · (u_norm · cos(r_L, x) · ‖x‖)
```

For `x ≈ r_L` (the canonical prompt), `cos(r_L, x) ≈ 1`, so `a_{i*}(x) ≈ σ(c_1) · c_2` for layer-typical scalar constants — a moderately large activation that fires the slot.

For `x = r_L'` from a different prompt (e.g. France's residual), `cos(r_L, r_L') ≈ 0.5–0.7` (residuals are correlated by template), and the activation is correspondingly suppressed. The suppression isn't strong enough at any single layer with α=5 to keep the new fact from breaking the old one (validated by alpha sweeps), but it is strong enough at `α=0.25` distributed across 8 layers.

### 6.3 Why multi-layer with small α works

Let `c = cos(r_L^{Atlantis}, r_L^{France})` ∈ [0.5, 0.7] (for "capital of …" prompts). Define

- `β_A^{(L)}(α) = a_{i*}(r_L^{Atlantis}) · ‖d_{i*}‖ ≈ k · α` per layer (some constant `k`)
- `β_F^{(L)}(α) ≈ k · α · σ(S · c) · c` — strictly less than `β_A^{(L)}(α)`

The competing Paris signal at France's L26 is approximately `β_P ≈ 80%` post-softmax weight, supplied by the trained features. Compounded across 8 layers:

- Total Atlantis push: `β_A^{total}(α) = 8 · k · α · 1` — needs to exceed Paris's signal at *Atlantis's* prompt (where Paris isn't a competitor) → α=0.25 gives 94.6% Pose.
- Total France push: `β_F^{total}(α) = 8 · k · α · σ(S · c) · c` — must be smaller than Paris's strong existing signal → at α=0.25 the perturbation is ~20% reduction in Paris probability (80.5% → 60.5%), Paris still rank 1.

A single layer at α=2.0 doesn't satisfy both because `c` is too high — the slot fires for both prompts indistinguishably. Multi-layer at small α uses the *cumulative* gap between cos=1 and cos≈0.6 across many independent attempts.

### 6.4 Refinement step (Gram-Schmidt against decoys)

When inserting many facts at the same layer (e.g. 10 different `(entity, X, target)` triples), the gates from earlier inserts contaminate later ones. The fix:

1. Cache *raw, pre-refine* residuals for every install: `R_L = { r_L^{(j)} : j ∈ inserts_at_L }`.
2. After each new insert at `L`, **rebuild every gate at `L` from the raw residuals** via modified Gram-Schmidt:

```
gate^{(j)} = unit( r_L^{(j)} - ∑_{k ≠ j} (r_L^{(j)}ᵀ ê_k) · ê_k - ∑_{decoy} (r_L^{(j)}ᵀ ê_d) · ê_d ) · g_norm · S        (13)
```

where `ê_k = unit(r_L^{(k)})` for peer install raws, and `ê_d` are decoy directions cached from canonical bleed-target prompts.

This is **batch refine** — recomputing from raws each time is idempotent. Online refine compounds drift: each iteration projects against already-refined peers, slowly walking off the right direction.

### 6.5 Storage and retrieval semantics

Two distinct mutation modes are supported:

- **`MODE COMPOSE`** (the math above) — synthesizes (gate, up, down) overrides that participate in inference through the standard FFN path. Compatible with COMPILE INTO MODEL.
- **`MODE KNN`** (default for one-line `INSERT`) — records a key vector `k = r̂_L` and a target string in `knn_store.bin`. At inference time, before logits, dispatch a KNN against `k` against active residuals; on a cosine match `> threshold`, force the target token. **Retrieval overlay, not a model edit.** Cannot be compiled into weights.

The KNN mode is faster to install and revertable, but it is *not* a mechanistic change — it's a sidecar lookup. The COMPOSE path is what the "model is the database" thesis rests on.

---

## 7. The COMPILE write-back

### 7.1 Vindex → Vindex (column rewrite)

For a patch with INSERT operations `{ (L_j, i_j, d_j) }` (down-vector overrides in the patched overlay), produce a fresh vindex where:

- All read-only weight files are **hardlinked** from source (instant on APFS / btrfs).
- A fresh `down_weights.bin` is built by:

```
for each layer L:
    slab = mmap(source_down_weights_at_layer(L))      # [h, m]
    for each (L_j, i_j, d_j) in patch where L_j == L:
        slab[:, i_j] = d_j                            # column splice
    write slab to output_down_weights.bin
```

Why `down_weights.bin` and not `gate_vectors.bin`? Because:

- The dense FFN reads down from `down_weights.bin` via `weight_manifest.json`.
- The gate row at the inserted slot is intentionally *not* written back to the file. The original (weak, near-zero) gate vector at that slot keeps the dense activation small. Combined with the strong down override (which has `S × g_norm × small_activation`-typical magnitude), the contribution at that slot during dense inference reproduces the patched session's contribution to within `f32 → f16 → f32` rounding.

End-to-end measured: the live patched session producing `Pose 56.16% / Paris 67.28%` round-trips through COMPILE → fresh `USE` to `Pose 56.91% / Paris 67.34%` — matching within rounding error.

### 7.2 Vindex → safetensors / GGUF (model export)

For `COMPILE INTO MODEL`:

1. Load the full weight set from the vindex.
2. For each install layer `L` with overrides:
   - Apply the gate / up / down values from the patch overlay directly into the in-memory `W_gate, W_up, W_down` tensors.
   - Optionally apply MEMIT closed-form weight editing as the elaboration step — solving `ΔW_down · K = R` with covariance regularization to make a *single layer* edit faithful when the multi-layer constellation is too disruptive.
3. Write tensors as standard safetensors using the architecture's tensor naming convention.
4. Copy `tokenizer.json` and configs.

The output loads in HuggingFace Transformers / vLLM / Ollama / llama.cpp without modification. The new fact is in the standard `down_proj` tensor; standard FFN code produces it.

---

## 8. The MEMIT formulation (for completeness)

For more sophisticated single-layer write-back, LARQL implements the **MEMIT** (Meng et al. 2022) closed-form weight editing solver. Notation: at install layer `L`, we want to add `Δ_{down}` to `W_down` such that for each fact `j`:

```
(W_down + Δ_{down}) · k^{(j)} = r^{(j)} + α · ê_T^{(j)}                                   (14)
```

where `k^{(j)} = activation_vector_at_L_for_prompt_j ∈ ℝ^m` (an "address"), and `r^{(j)} = W_down · k^{(j)}` is the original output. Minimum-norm `Δ` solving (14) over multiple `(k, target_delta)` pairs is the Tikhonov-regularized least squares:

```
Δ_{down} = (T - W_down · K) · (K · Cᵀ · K + λI)⁻¹ · K                                     (15)
```

with

- `K = [k^{(1)}, …, k^{(N)}] ∈ ℝ^{m × N}`,
- `T = [α·ê_T^{(1)}, …, α·ê_T^{(N)}] ∈ ℝ^{h × N}`,
- `C = E_x [ a(x) a(x)ᵀ ]` — the FFN activation covariance, estimated from a diverse prompt set.
- `λ` — regularization coefficient.

This is the sophistication that's available when you want a single-layer commit instead of the multi-layer constellation. It produces an exact closed-form update; the constellation is the engineering shortcut that doesn't need covariance estimation and works at install-time without an offline step.

---

## 9. Residual stream additivity (TRACE)

The trace decomposition treats the residual stream as a single bus written by every block:

```
x^{(0)} = embed(token)
x^{(L)} = x^{(L-1)} + δ_attn^{(L)}(x^{(L-1)}) + δ_ffn^{(L)}(x^{(L-1)} + δ_attn^{(L)})    (16)
```

Define `attn_delta^{(L)} = δ_attn^{(L)}(x^{(L-1)})` and `ffn_delta^{(L)} = δ_ffn^{(L)}(…)` after attention. Then:

```
x^{(L)} = x^{(L-1)} + attn_delta^{(L)} + ffn_delta^{(L)}                                  (17)
```

Reconstruction is **exact** by addition. A trace stores `attn_delta` and `ffn_delta` for every layer; `x^{(L)}` recovers via summation.

For each token `t` in the vocabulary, project `x^{(L)}` through the (norm-then-)unembed:

```
logit^{(L)}(t) = lm_head_t · norm(x^{(L)})                                                (18)
prob^{(L)}(t) = softmax(logit^{(L)})_t                                                    (19)
```

The trace's `answer_trajectory(t)` plots `prob^{(L)}(t)` and the per-layer contributions of attn vs ffn:

```
Δlogit_attn^{(L)} = lm_head_t · norm(x^{(L-1)} + attn_delta^{(L)}) - lm_head_t · norm(x^{(L-1)})
Δlogit_ffn^{(L)}  = lm_head_t · norm(x^{(L)}) - lm_head_t · norm(x^{(L-1)} + attn_delta^{(L)})
```

Each layer's attn/ffn split tells you which mechanism is doing the work for the answer. The Gemma 3 4B trace for "The capital of France is" shows:

| Layer | Rank | Prob | Δattn(Paris) | Δffn(Paris) | Who |
|------:|-----:|-----:|-------------:|------------:|-----|
| L22 | 50 | 0.002 | +22.2 | +34.4 | both |
| L23 | 10 | 0.024 | -16.9 | +55.9 | FFN |
| **L24** | **1** | **0.714** | **+105.7** | **+24.4** | **both** ← phase transition |
| L25 | 1 | 0.997 | +4.3 | +94.4 | FFN |
| L26 | 1 | 0.999 | +83.1 | +18.7 | both |

The phase transition at L24 is *visible* — it's the 105-point attention spike. This is why the trace storage is interesting: it makes mechanistic claims about a forward pass *checkable*.

---

## 10. Putting it all together: a forward pass through the graph

Here is the complete forward pass viewed entirely as graph operations and additive residual writes. Below, "node" = current point in residual space, "edge" = FFN feature, "router" = attention block.

```
node = embed(token_seq)                                  # [seq, h]
for L in 0 .. num_layers-1:
    # Attention: routes information across positions.
    # No graph operation here — pure dense matmul + softmax.
    node += attention^{(L)}(norm(node))

    # FFN: graph traversal.
    pre_ffn = norm(node)                                 # post-attention residual
    for s in 0 .. seq_len:
        x_s = pre_ffn[s, :]
        # Step 1: compute active set via gate KNN.
        A = top_K_indices( |W_gate^{(L)} · x_s| )
        # Step 2: aggregate edge contributions.
        out_s = 0
        for i in A:
            a_i = σ(g_iᵀ x_s) · (u_iᵀ x_s)                 # activation
            out_s += a_i · d_i                            # add the edge's payload
        node[s, :] += out_s

# Final unembed.
logits = lm_head · norm(node[-1, :])
return softmax(logits)
```

**Every per-token operation is a graph traversal** — find the active edges, sum their weighted contributions. The attention block remains a separate dense computation (it's a router, not a knowledge store; converting it to a graph is the subject of OV/RD experiments and is beyond this paper's scope).

The complexity per token, per layer is:

- Gate KNN: `O(m · h)` (one BLAS gemv).
- Sparse aggregate: `O(K · h)` (K up dots + K down adds-with-coefficient).

For Gemma 3 4B with `K=8092` (close to `m=10240`), this is barely cheaper than dense. The win comes when `K << m`:

- Future K-quant FFN where the dense matmul is bandwidth-bound.
- MoE models where features are sharded across experts (effective `K_effective = K_per_expert · top_k`).
- Sparse activation regimes where the model has been trained for `K = 256` or smaller (TopK FFN, Switch Transformer style).

For LARQL today, **the equivalence is the primary product**: walk-FFN proves the matrix multiplications can be replaced by a graph traversal without changing the output. The performance argument is secondary and already pays off for mmap layouts.

---

## 11. Summary of identities

For reference:

| Identity | Statement |
|----------|-----------|
| **Matmul = graph sum** | `δ_ffn(x) = ∑_i a_i(x) · d_i` exactly |
| **Walk = dense at full K** | `δ̃_ffn^{(K=m)}(x) = δ_ffn(x)` |
| **Sparsity bound** | `‖δ - δ̃‖ ≤ √(∑_{i ∉ A_K} a_i² · ‖d_i‖²)` |
| **Gate KNN = MIPS** | `A_K(x) = argmax-K_i |g_iᵀ x|` |
| **Circuit type** | `cos(g_i, d_i)` partitions edges into projector / identity / transform / inverter / suppressor |
| **Constellation insert** | `g_{i*} = r̂_L · g_norm · S`, `u_{i*} = r̂_L · u_norm`, `d_{i*} = ê_T · d_norm · α`, summed over L install layers |
| **Refinement** | Modified Gram-Schmidt of new gate against (peer raws ∪ decoys) |
| **COMPILE round-trip** | `down_weights.bin[:, i_j] := d_j` for each install slot; gate vectors *not* baked back |
| **Additive trace** | `x^{(L)} = x^{(L-1)} + attn_delta^{(L)} + ffn_delta^{(L)}` |

These identities are sufficient to re-implement the LARQL pipeline from scratch in any language. The next document, `03-software-architecture.md`, describes how LARQL itself organizes the corresponding code.
