# LARQL: Mathematical Foundation

> A self-contained mathematical treatment of the LARQL graph-decompilation
> technique, sufficient as a basis for new implementations and for reasoning
> about correctness and limitations.

## Notation

| Symbol | Meaning |
|--------|---------|
| `H` | hidden size of the model |
| `I` | intermediate (FFN) size; `I ≫ H` typically |
| `V` | vocabulary size |
| `L` | number of transformer layers |
| `x ∈ ℝ^H` | a residual-stream vector at one token position |
| `E ∈ ℝ^{V×H}` | token embedding matrix (rows are token vectors) |
| `s ∈ {0,…,V−1}` | a token id |
| `e_s = E[s, :] ∈ ℝ^H` | embedding of token `s` |
| `W_g, W_u ∈ ℝ^{I×H}` | FFN gate and up-projection matrices, per layer |
| `W_d ∈ ℝ^{H×I}` | FFN down-projection matrix, per layer |
| `g_i := W_g[i, :] ∈ ℝ^H` | gate row of feature `i` (the *trigger*) |
| `u_i := W_u[i, :] ∈ ℝ^H` | up row of feature `i` (the *gain*) |
| `d_i := W_d[:, i] ∈ ℝ^H` | down column of feature `i` (the *write*) |
| `σ` | the FFN activation function (SiLU/GeGLU etc.) |
| `K` | number of features kept by gate-KNN per layer |
| `α` | per-layer strength of an inserted edge |

We treat `W_g`, `W_u`, `W_d` as carrying an implicit layer index `ℓ` when
needed, and we work with one token position at a time; the vectorisation
across positions is straightforward.

---

## 1. The FFN as an explicit edge sum

The standard gated FFN block is

```
y = W_d ( σ(W_g x) ⊙ (W_u x) )                                    (1)
```

The product inside the parentheses is element-wise. Equation (1) is
mathematically identical to the explicit sum

```
y = Σ_{i=0}^{I−1}  σ(g_i · x) · (u_i · x) · d_i                   (2)
                     └────┬───┘   └───┬───┘
                       gate scalar  up scalar
                     └─────────────┬─────────────┘
                                a_i := activation of feature i
```

Each term in the sum is a contribution from a single feature:

- a scalar `a_i := σ(g_i · x) · (u_i · x)`,
- multiplying a fixed direction `d_i ∈ ℝ^H` (the down column).

This is the **graph reading** of the FFN. Each feature `i` is an
edge whose:

- *trigger* is the gate row `g_i` (the input pattern it detects),
- *gain* is the up row `u_i` (an amplitude correction),
- *write* is the down column `d_i` (the direction it adds to `x`).

For ungated FFN (e.g., GPT-2's `down(GELU(up(x)))`), the same decomposition
holds with `a_i = σ(u_i · x)` and no gate row — the trigger is `u_i`. In
gated architectures the gate provides sharper selectivity and is the
natural index.

---

## 2. The walk: top-K approximation of (2)

Because `σ(g_i · x)` is sharply peaked (in particular, SiLU is bounded
below by ~−0.28 and pushes most negative pre-activations toward 0), the
sum (2) is empirically dominated by a small number of features `i` per
layer per token. Define the top-K set

```
T_K(x) := argmax^{(K)}_{i ∈ {0,…,I−1}} | σ(g_i · x) · (u_i · x) |
                                       └────────┬─────────┘
                                              |a_i|
```

and the walk approximation

```
y_K  :=  Σ_{i ∈ T_K(x)}  a_i · d_i                                (3)
```

**Claim (empirically verified, walk-boundary sweep on Gemma 3-4B):** for
appropriately small `K` (e.g. 10–100 per layer), `y_K = y` to within
floating-point noise across all 34 layer boundaries, with identical top-1
predictions and matching top-K probability orderings. See
[`docs/walk-boundary-sweep.md`](../walk-boundary-sweep.md) for the
empirical validation across all `L0…L34` boundaries.

This is the central computational claim of LARQL: the FFN, although
written as a dense `H × I` matrix multiply, *behaves as a sparse
dispatch* into a small constellation of features per token.

### 2.1. Computing the top-K efficiently — gate KNN

The brute-force computation of `T_K(x)` requires `I` dot products of
length `H`, i.e. `O(IH)` — exactly the cost of the gate matmul. The
LARQL implementation does not pre-emptively shortcut this scan; instead
it stores the gate matrix as a contiguous mmap, batches the dot products
through BLAS, and selects the top-K by partial sort:

```
scores  ←  W_g x          # (I,) via gemv
T_K     ←  partial_argmax(|scores|, K)
```

The point is not that the cost is lower in FLOPs (it isn't), but that
the *materialised set* `T_K(x)` together with its scores is now an
explicit, queryable, editable object. From it we cheaply compute

```
a_i  =  σ(scores[i]) · (u_i · x)        for i ∈ T_K            (4)
y_K  =  Σ_{i ∈ T_K}  a_i · d_i                                  (5)
```

so the second half of the FFN — the down projection — is a sparse
gather of `K` columns of `W_d` rather than a dense `H × I` gemm. When
`d_i` is stored in a feature-major mmap'd file, this gather has
favourable cache behaviour and on Gemma 3-4B the walk forward beats
the dense forward by a few percent on a single CPU.

### 2.2. Why `|a_i|` and not `a_i`

Negative contributions matter. Some features write *against* a direction
to suppress an alternative. Selecting by absolute value preserves both
positive (additive) and negative (subtractive) features — the experiments
in `circuit-types` show that "inverter" features (cos(g_i, d_i) < −0.5)
are functionally important and would be missed by selecting positives only.

---

## 3. Edge labelling — making features queryable as facts

Equation (3) expresses the FFN's computation as a sum over features.
LARQL labels each feature by projecting its trigger and write through
the embedding table.

### 3.1. The trigger-side label

For feature `i` at layer `ℓ`, its **input affinity** is

```
c_in(i, ℓ)  :=  max_{s ∈ V}  ( e_s · g_i )                       (6)
ŝ_in(i, ℓ)  :=  argmax_{s ∈ V}  ( e_s · g_i )
```

`ŝ_in` is the token whose embedding is most aligned with the gate. The
top-K version of (6) gives a multi-language label (cf. cross-lingual
results in `findings.md`: a single feature's gate often matches
"French / french / Frenchman / француз / フランス" simultaneously).

### 3.2. The write-side label

Symmetrically, the **output affinity** is

```
c_out(i, ℓ)  :=  max_{s ∈ V}  ( d_i · e_s )                      (7)
ŝ_out(i, ℓ) :=  argmax_{s ∈ V}  ( d_i · e_s )
```

This is the down-KNN that recovers labels like

```
F5040: down-KNN → French, French, FRENCH, France, Frenchman, ...
F943:  down-KNN → euros, €, EU, Euros, Spain, EUR, ...
```

### 3.3. Confidence and selectivity

Per layer, normalise the joint product:

```
c(i, ℓ)         :=  c_in(i, ℓ) · c_out(i, ℓ) / max_j (c_in(j, ℓ) · c_out(j, ℓ))   (8)
selectivity(i)  :=  c_in(i, ℓ) / max_j  c_in(j, ℓ)                                 (9)
```

`c` is the LARQL "confidence" score (0 to 1). `selectivity` measures how
specific the trigger is to a single embedding. Empirically:

- `c` peaks at early layers (L6–L12) — strong combined signals there are
  *structural* (morphology, syntax, articles).
- `selectivity` peaks at late layers (L25–L33) — strong selectivity there
  is *factual*.

So filtering depends on what you want:

- Factual edges:   `selectivity ≥ 0.15` and `ℓ ∈ [25, 33]`
- Structural edges: `c ≥ 0.5` and `ℓ ∈ [0, 14]`

The labelled triple `(ŝ_in, ℓ-F<i>, ŝ_out)` together with `(c, selectivity)`
defines an edge in the LARQL knowledge graph.

---

## 4. The residual-stream identity

The transformer's forward pass at one token can be written as a
running sum:

```
x^{(0)}    = embed(token)
x^{(ℓ)}    = x^{(ℓ−1)} + Δ^{(ℓ)}_attn + Δ^{(ℓ)}_ffn               (10)
logits     = LMHead(x^{(L)})
```

where each `Δ` is the residual contribution of that block at that layer.
Equation (10) is *additive*: layers add to a running residual; nothing
is overwritten. This exact additivity is what makes the trace
decomposition and the multi-layer insert work.

### 4.1. Trace decomposition

For any prompt and target token `t`, define the per-layer logit
contribution to `t`:

```
log_t^{(ℓ)}     =  e_t · x^{(ℓ)} / scale                                      (11)
attn_push(ℓ, t) =  e_t · Δ^{(ℓ)}_attn / scale                                 (12)
ffn_push(ℓ, t)  =  e_t · Δ^{(ℓ)}_ffn  / scale                                 (13)
```

with `log_t^{(ℓ)} = log_t^{(ℓ−1)} + attn_push(ℓ, t) + ffn_push(ℓ, t)`.
The TRACE statement reports these three numbers per layer, recovering
exactly where in the depth dimension the answer crystallises. The
Gemma 3-4B trace for "the capital of France is" shows a clean phase
transition at `L24` (attention pushes +106) and `L25` (FFN pushes +94),
with both blocks reinforcing each other from `L26` onward.

### 4.2. Two-stroke pattern

Empirically across many factual prompts:

- attention dominates at *even* late layers (L24, L26, …) — it routes
  the right prior-token information into position;
- FFN dominates at *odd* late layers (L23, L25, L27, …) — it writes the
  factual answer.

This is the "two-stroke engine" picture and follows directly from the
additive residual identity (10).

---

## 5. Training-free knowledge insertion

The goal of insertion is to choose new feature parameters
`(g, u, d)` and a slot `s` (one per "constellation" layer) such that, on
a target prompt, the model's forward pass resolves to a desired token
without retraining. We make the assumption that we have one or a few
free slots per layer, i.e. slots whose existing `c_in × c_out` product
is near zero — empirically there are many such slots in trained
transformers.

### 5.1. The wrong gate

Suppose we want to insert `(Atlantis, capital-of, Poseidon)` at layer
`ℓ`. The naive choice `g ← e_{Atlantis}` fails because

```
cos( e_{Atlantis}, x^{(ℓ)}_{Atlantis prompt} )  ≈  0.01
```

(measured on Gemma 3-4B at `ℓ = 24`). After 24 layers of attention and
FFN the residual is essentially orthogonal to the raw token embedding —
they live in very different subspaces of `ℝ^H`. The gate would never
fire.

### 5.2. The right gate — captured residuals

Instead, run *one forward pass* on the trigger prompt, capturing
`x^{(ℓ)}` at each chosen layer. Use that residual as the new gate, scaled
to the magnitude of the existing gates so the activation distribution
matches:

```
ḡ_ℓ := mean_i ‖g_i^{(ℓ)}‖                                       (14)
g_new^{(ℓ)} := x^{(ℓ)} · ḡ_ℓ / ‖x^{(ℓ)}‖                        (15)
```

By construction `g_new · x^{(ℓ)} = ḡ_ℓ · ‖x^{(ℓ)}‖`, which is large; the
new feature fires strongly on this prompt. Crucially, *neighbouring*
prompts (e.g. "the capital of France is") have residuals at `ℓ` that
differ from `x^{(ℓ)}_{Atlantis}` by a small angle (cos ≈ 0.98 across
prompts that share a template), so the gate also fires for them — but
proportionally less strongly.

### 5.3. The down vector — embedding direction

The LM head of most modern transformers is tied to the embedding,
`LMHead = E^⊤`. Therefore the logit of token `t` is
`(E x^{(L)})_t = e_t · x^{(L)}` (modulo a final scale). To increase the
logit of `Poseidon` we must add `e_{Poseidon}` (scaled appropriately) to
the residual at some layer:

```
d_new^{(ℓ)} := e_{Poseidon} · embed_scale · α                   (16)
```

with `embed_scale = √H` to match the residual-stream scale, and `α` a
small per-layer strength.

### 5.4. The constellation — multi-layer spread

Single-layer insertion at large `α` succeeds on the target prompt but
breaks neighbours: at the strength needed to push Atlantis to Poseidon,
France also pushes to Poseidon (because the gates cos-correlate at 0.98
across prompts). Single-layer at small `α` fails to move the target.

The constellation insert spreads the change across `n` layers in the
upper knowledge band, each at strength `α/n`:

```
For ℓ ∈ {ℓ_0, ℓ_0+1, …, ℓ_0+n−1}:
   choose a free slot s_ℓ
   write at slot s_ℓ:
       g_new^{(ℓ)} :=  x^{(ℓ)} · ḡ_ℓ / ‖x^{(ℓ)}‖
       u_new^{(ℓ)} :=  same direction, scaled to ū_ℓ
       d_new^{(ℓ)} :=  e_target · embed_scale · α               (17)
```

This is exactly the `install_edge` primitive in
[`crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs:53`][edge].

[edge]: ../../crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs

The "magic" of why the constellation works without breaking neighbours is
purely arithmetic. Consider the residual contribution at the LM head from
the inserted constellation, on prompt `p`:

```
Σ_{ℓ}  σ(g_new^{(ℓ)} · x^{(ℓ)}_p) · (u_new^{(ℓ)} · x^{(ℓ)}_p) · d_new^{(ℓ)}
= ( Σ_{ℓ}  a^{(ℓ)}(p)  · α )  ·  (e_target · embed_scale)              (18)
```

where `a^{(ℓ)}(p)` is the activation of the new feature on prompt `p` at
layer `ℓ`. The change in the target-token logit is therefore
proportional to the *layer-summed activation* on `p`.

For the trigger prompt `p★` (the prompt the gate was captured from):
`a^{(ℓ)}(p★)` is large at every `ℓ`, so the sum scales nearly linearly
with `n`.

For a near-neighbour prompt `q`:
`a^{(ℓ)}(q)` is also positive (because the residuals share template
structure) but each individual activation is *competing with the
existing strong signal* (Paris) from the trained features. The network's
softmax dampens incremental pushes when the leader has a large margin —
each `α/n` nudge from the constellation is partially absorbed by the
incumbent's own subsequent layers reinforcing it. So the cumulative
effect on `q` grows *sub-linearly* in `n`.

The Pareto frontier observed empirically (8 layers × `α`=0.25, or
16 layers × `α`=0.12) reflects exactly this trade-off: spread thinner
to reduce neighbour damage, spread thicker to maximise target
confidence.

### 5.5. Norm preservation in `install_edge`

To keep the inserted feature in the same magnitude regime as trained
features (and so behave consistently inside the model's later layers),
`install_edge` computes the original slot's norms and scales the trigger
and write to match:

```
g_scale  :=  g_norm  · gate_scale  / ‖trigger‖
u_scale  :=  u_norm  / ‖trigger‖
α_eff    :=  (d_norm / ‖write‖) · α_mul

gate[s, :]  :=  trigger * g_scale
up[s,   :]  :=  trigger * u_scale
down[:, s]  :=  write   * α_eff                                          (19)
```

`gate_scale` (typically 30) makes the gate fire decisively when the
trigger appears (well above the typical activation of trained features);
`α_mul` is the per-layer strength `α` from §5.4. Both `g_norm`, `u_norm`,
`d_norm` are read off the original slot, so even when "the original
slot" is essentially empty the inserted feature inherits the model's
own scale conventions.

---

## 6. Compile-time round-trip

After a sequence of inserts, deletes, and updates, the LARQL graph holds
overrides at chosen `(ℓ, i)` pairs. `COMPILE INTO MODEL` materialises
each override into the dense weight matrices via (19), then writes
`safetensors`/`GGUF`. Because the only changes are at chosen FFN slots,
and those slots were chosen from features that were near-zero in the
original model, the rest of the dense matrices are unchanged.

A useful invariant: the `COMPILE` operation is a left-inverse of
`EXTRACT`. Specifically, for an unedited vindex,

```
EXTRACT → COMPILE  =  identity (up to floating-point round-trip)
```

(file-level: with f16 quantisation the round-trip is approximate; with
f32 it is bit-exact for the FFN block.) After edits, the identity
becomes

```
EXTRACT → INSERT(...)k → COMPILE  =  original_weights + Σk insertion_overlay_k
```

where each `insertion_overlay_k` is non-zero only in the slots written
by `install_edge`.

---

## 7. Correctness and limitations

### 7.1. Walk correctness

The walk approximation (3) replaces the exact sum (2) by its top-K
truncation. The error is bounded by the L1-norm of the omitted
activations:

```
‖y − y_K‖   ≤   Σ_{i ∉ T_K(x)}  |a_i|  ·  ‖d_i‖                  (20)
```

The empirical observation is that in trained transformers this bound is
tiny when `K = 10–100` and `I = 10,240` (Gemma 3-4B). This is a
consequence of the heavy-tailed distribution of `|a_i|` and is not
proven in general — it must be verified per architecture per family.
The walk-boundary sweep in [`docs/walk-boundary-sweep.md`](../walk-boundary-sweep.md)
reports zero top-1 divergence at every Gemma-3-4B layer boundary.

### 7.2. Insertion limitations

The insert technique (§5) has known limitations:

- **Subtoken targets.** Inserting `Poseidon` produces strong logit only
  on the first BPE subtoken `"Pose"`; "idon" must come from the
  autoregressive continuation, which is not biased by the constellation.
- **Neighbour degradation.** At the recommended 8L × 0.25 setting,
  France→Paris drops from ≈80.5% to ≈60.5% — a 20-pt collateral cost.
  Using more layers at smaller `α` (16L × 0.12) reduces this to ~14 pt
  but at the price of the target only reaching ~78% confidence.
- **Non-selective firing.** The constellation gate fires on any prompt
  whose residual at the chosen layers is template-correlated with the
  trigger prompt. It is not a per-entity precision instrument; it is a
  per-template instrument.
- **Calibration is per-model.** `embed_scale = √H` is a Gemma/Llama
  convention; other architectures (untied LM heads, different RMSNorm
  scales) require recalibration of (16) and (19).

### 7.3. Architectural assumptions

The whole framework assumes:

- A *gated* FFN (or, with mild adaptation, an ungated one with `up` as
  the trigger), so features have a clear scalar activation per token.
- *Residual* connectivity — equation (10) — so contributions are
  additive and decomposable.
- *Tied or near-tied* LM head, so equation (16) places the down vector
  in the right direction. With an untied head a corresponding head row
  must be substituted.

These hold for all of the families enumerated in `README.md`'s "Model
Support" table; they would fail for, e.g., RWKV-style state-space
models where the residual identity (10) does not hold layer-wise.

---

## 8. Summary of the core equations

The technique can be reduced to seven equations:

```
(2)  y          = Σ_i  σ(g_i · x) · (u_i · x) · d_i              (FFN as edge sum)
(3)  y_K        = Σ_{i ∈ T_K(x)}  σ(g_i · x) · (u_i · x) · d_i   (walk = top-K truncation)
(8)  c          = c_in · c_out / max_j(c_in · c_out)             (edge confidence)
(10) x^{(ℓ)}    = x^{(ℓ−1)} + Δ_attn + Δ_ffn                     (residual identity)
(15) g_new      = x^{(ℓ)} · ḡ / ‖x^{(ℓ)}‖                        (gate from captured residual)
(17) Σ_ℓ d_new  = Σ_ℓ  e_target · embed_scale · α                (constellation write)
(19) install_edge primitive (gate/up/down assignment with norm preservation)
```

All higher-level operations (`DESCRIBE`, `WALK`, `INFER`, `INSERT`,
`COMPILE`, `TRACE`) are implementable as compositions of these
primitives plus standard transformer attention. There is no hidden
machinery.

---

## 9. Open mathematical questions

A short, non-exhaustive list:

1. **Provable bounds on `K`.** Under what regularity assumptions on
   trained `W_g, W_u, W_d` is the walk truncation (3) provably
   `ε`-accurate for `K = O(polylog(I))`? Empirically `K ≪ I` works;
   theoretically the heavy-tail behaviour of feature activations is
   not yet quantified.

2. **The dark space.** 85% of features in Gemma-3-4B have down vectors
   that do not align well with any single embedding (`c_out` near
   noise floor). Are these features structurally necessary — i.e.,
   is there a quantitative measure of "structural" vs. "factual"
   that recovers this 15/85 split from first principles, rather than
   from observed `c_out` alignment? Connections to feature
   superposition literature are obvious but not yet formalised here.

3. **Insertion capacity.** What is the maximum number of facts that
   can be inserted via constellations at fixed neighbour-degradation
   budget? Empirically 10/10 retrieval at 5–10 facts (exp-14); is
   there a sub-linear capacity limit, and does it correlate with the
   manifold dimensionality of the residual stream (exp-02 SVD)?

4. **Higher-order edges.** All of §3 treats features as
   single-input/single-output. Some experiments find features that
   bridge two inputs to one output (relations) or one input to two
   outputs (sub-token writes). A graph-theoretic generalisation
   beyond simple directed edges would extend the framework.

5. **Attention as a graph.** The current LARQL FFN graph is
   complete-but-unindexed; attention is the missing index. The QK
   factorisation `W_Q W_K^⊤` admits SVD-based decomposition and could
   be expressed as a (head-aware) graph of "this token attends to
   tokens of type T at distance D." Experiments 16, 18, 19, 24, 38
   pursue this; a unified attention-as-graph mathematics is open.

These questions do not affect the practical correctness of the
existing implementation but determine how far the framework can scale
and how rigorously its claims can be made.
