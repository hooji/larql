# LARQL — Executive Summary

> A 2-page brief on a technique that decompiles transformer model weights into a queryable graph database, allows direct edits without retraining, and recompiles them back to standard model formats. Intended for executives evaluating impact and strategic positioning.

---

## What it is

LARQL ("Lazarus Query Language") is an open-source toolchain (Apache-2.0, ~1.7k Rust files in production, 2,000+ tests) that takes a trained transformer model — Gemma, Llama, Mistral, Mixtral, Qwen, Phi, DeepSeek, GPT-OSS, GPT-2 — and reorganizes its weights into a directory of memory-mapped files called a **vindex**. The vindex exposes the model as a graph: each FFN feature is a labeled edge (gate vector = trigger, down vector = output), so the entire 8-billion-parameter model becomes a 348,160-edge graph that can be browsed, queried, and edited from the command line.

A SQL-like query language, **LQL**, runs over the vindex:

```sql
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex";
DESCRIBE "France";          -- shows the model's knowledge of France
INFER "The capital of Atlantis is" TOP 5;
INSERT INTO EDGES (entity, relation, target)
   VALUES ("Atlantis", "capital-of", "Poseidon");
COMPILE CURRENT INTO MODEL "edited/" FORMAT safetensors;
```

The end-to-end demonstration: a single new fact is inserted in ~30 ms (one forward pass plus eight feature writes, no GPU), the model thereafter answers `"The capital of Atlantis is"` with `"Poseidon"` at 94.6% confidence, the existing answer for France is preserved, and the result can be exported to a standard HuggingFace safetensors file that loads in PyTorch / vLLM / Ollama / llama.cpp without modification.

---

## What's strategically novel

Three claims that no incumbent system makes:

1. **The trained model already is a graph database.** A FFN's gate row is the "address" and its down column is the "value"; the matrix multiplication that runs the forward pass is mathematically identical to a graph traversal that picks the active features and sums their weighted contributions. LARQL is the first system to take this seriously enough to build a query language and storage format around it.

2. **Knowledge can be edited in seconds without training.** Inserting a fact is a single forward pass to capture residuals, plus eight scalar writes synthesizing (gate, up, down) vectors that don't disturb neighbouring facts. No backprop. No GPU. No fine-tuning loop. Validated end-to-end at 94.6% new-fact confidence with 60.5% preservation of the most-related existing fact.

3. **Edits round-trip back to standard model formats.** A `.vlp` patch (typically ~30 KB per fact) is JSON. `COMPILE INTO MODEL` writes a fresh safetensors directory; the new fact is in the standard `down_proj` tensors and triggers through the standard FFN path with no special loader. The model file remains a model file.

These three pieces — graph view of the weights, training-free edit, round-trip COMPILE — are the work's core differentiator. The accompanying engineering (mmap-everywhere storage, hardlink-based COMPILE, distributed CPU MoE serving via `larql serve --experts 0-63`, Metal GPU at 19 tok/s on Gemma 4 26B-A4B, walk-FFN measured *faster* than dense matmul) is what turns the research idea into something a developer can `cargo install` and use.

---

## What it changes for consumers of LLM services

- **Local-first knowledge editing.** A developer who needs to teach a model 1,000 organisation-specific facts (employee directory, product catalog, internal acronyms) does not need a fine-tuning pipeline, a GPU rental budget, or weeks of iteration. They write 1,000 INSERT statements (~30 MB total patch) and COMPILE; the result is a 8 GB safetensors file ready for any inference runtime. The cost is the cost of a `git push`.
- **Inspectable behaviour.** `DESCRIBE "Paris"` returns the model's actual associations, sourced to specific layers and features. When a model hallucinates, an operator can ask it *why* and remove the offending edge with `DELETE FROM EDGES WHERE …`.
- **Smaller deployment footprint.** The vindex format separates weights by intent: a 4B model serves browse-only at 3 GB, full inference at 6 GB, or carved into a 7 GB "client" slice that runs on a laptop while the FFN lives on a CPU server. A Kimi K2.6 / DeepSeek V4-class trillion-parameter model becomes a multi-shard CPU deployment instead of an H100 cluster.
- **Privacy and provenance.** Edits, deletions, and patches are auditable JSON. Removing a memorized PII string is a `DELETE` statement; the deletion is recorded in a `.vlp` file you can share or version-control. There is no opaque retraining cycle; every byte change is attributable.

---

## What it changes for large AI labs (OpenAI, Anthropic, Google, Meta)

Three workflows are directly affected:

- **Surgical knowledge updates.** When a customer reports an error like "the model thinks the iPhone 14 has a USB-C port" or "the model insists that the company HQ is in Boston when it's actually New York", today's recourse is a fine-tune cycle, a RAG sidecar, or a system-prompt patch. With LARQL-style editing, the fix is a single INSERT that ships in the next minor model release with no retraining. The cost of the *N*-th fact-fix scales linearly in *N*, not in retraining time.
- **Continuous knowledge currency.** A model's "knowledge cutoff" is a hard constraint today because retraining is expensive. With training-free insertion, a daily / weekly knowledge update from a reliable source (Wikipedia diffs, news summaries, financial data) becomes a `cron` job that produces a `.vlp` patch, applies it, and ships a new model snapshot. The cutoff conversation changes from "January 2026" to "yesterday".
- **Mechanistic interpretability surface as a product.** LARQL's mech-interp hooks (capture, ablate, steer, activation patching, logit lens, KV-cache surgery, residual trace decomposition) are competitive with `nnsight` / `TransformerLens` / `chuk-mcp-lazarus`. A safety / interpretability team gets these for free against any model that can be extracted to a vindex. The same primitives drive customer-facing tools (red-teaming, jailbreak detection, refusal probes) that are currently bespoke per model.

There are also second-order strategic implications worth thinking about:

- **Distillation of the moat.** If editing is free and recompilation is round-trippable, the value of a particular base model becomes more about its quality at training time and less about exclusive access to its weights. A model that is open-weights *and* easily editable becomes a better starting point for an enterprise than a closed model behind an API. LARQL strengthens the position of open-weights families (Llama, Gemma, Qwen) at the expense of closed APIs.
- **Litigation surface.** "The model said X about my client" is currently a difficult complaint to resolve because evidence and remediation paths are opaque. A graph database of the model's facts, with `DESCRIBE` showing exactly which features fire and `DELETE` providing direct remediation, is a more defensible posture for a model provider.
- **Compliance.** EU AI Act / GDPR right-to-be-forgotten requests against trained models are technically intractable today. A `DELETE FROM EDGES WHERE entity = "John Doe"` followed by a `COMPILE` is at least a credible attempt.

---

## What it changes for the open-source AI community

- **Model interpretation becomes routine.** The current state of art for "what does this model know about X" is to write Python against `transformer_lens`, run a custom probe pipeline, and interpret the output. LARQL collapses this to a one-line LQL statement against a pre-built vindex. The HuggingFace publishing flow already supports vindex artifacts: `larql publish` uploads the full vindex plus six sliced siblings (browse / client / server / attn / embed / expert-server) and three nested collections (model / family / library) in one command.
- **A common substrate for "edit a model" research.** ROME, MEMIT, MEND, model surgery — most knowledge-editing research today re-implements a basic capture / write / probe loop. LARQL ships that as `LayerHook` + `INSERT INTO EDGES` + `TRACE … FOR target`. New methods can plug in as `MODE COMPOSE` extensions with shared infrastructure for capture, decoy management, refinement, and verification.
- **CPU-only deployment unlocks new audiences.** The Metal GPU path is the fastest, but Linux/Windows/x86 also work via OpenBLAS. The `serve --experts 0-63` topology means a researcher with a small budget can host a 26B-A4B MoE model on a couple of cheap CPU VMs (tested on `fly.io`, see `deploy/fly/`). The barrier to "I have my own LLM running" drops from "I need an H100" to "I need a $20/month VM".
- **A reference for what "graph view of a transformer" should look like.** The spec documents (`vindex-format-spec.md`, `vindex-operations-spec.md`, `vindex-ecosystem-spec.md`, `lql-spec.md`, `trace-format-spec.md`) are detailed enough that a third party can re-implement the format. The Apache-2.0 license + the published HuggingFace vindexes mean derivative work is unblocked.

---

## What is uncertain

The technique is real and shipping, but several dimensions are still being measured:

- **Quality of multi-fact edits at scale.** 10-fact INSERT works (`refine_demo` shows 10/10 retrieval). 1,000-fact INSERT is plausible but not extensively measured; the Gram-Schmidt refinement is `O(N^2)` in the per-layer install set and may need bucketing.
- **Behaviour on closed-weights models.** LARQL only works on accessible weights. Closed APIs (GPT-4, Claude, Gemini production endpoints) are not extractable. The technique is a force multiplier for open-weights models, not a way to edit closed ones.
- **MXFP4-quantized MoE (GPT-OSS) browse quality.** The 4-bit weight precision produces noisy gate-KNN dot products; DESCRIBE/WALK on GPT-OSS-120B is degraded. INFER works at full quality. Restoration paths are documented (residual-based DESCRIBE, gated-KNN with up activation, full-precision re-extract).
- **Attention is still a dense matmul.** The graph view applies cleanly to FFN. Attention remains the four Q/K/V/O projections plus softmax. Experiments (16, 17, 18, 38) prototype an attention-as-graph view, but it is not in production.

---

## The bottom line

LARQL operationalizes a particular reframing of how transformer FFNs work: not as a black-box matrix multiplication, but as a content-addressable graph database whose edges are the (gate, down) vector pairs. Once you accept that view, three previously-hard things become routine: querying what the model knows, editing it without retraining, and round-tripping the edits back to standard model files.

The technique is real, the code is shipping, the demonstrations work end-to-end on production models (Gemma 3/4, Llama 2/3, Mixtral, Qwen, Phi, DeepSeek, GPT-OSS, GPT-2). It targets open-weights models — there is no path to editing closed-API models. For organizations whose strategy depends on either open-weights customization (most enterprises, most open-source consumers) or transparency / interpretability of model behaviour (regulated industries, safety teams, mech-interp research), the toolchain is a meaningful new capability.

The biggest implication is structural: **a model becomes a thing you can read, edit, and ship as a diff**, on the same day, by hand, without retraining. That is qualitatively different from today's "fine-tune or RAG" choice and likely reshapes what "iterating on a model" means in production.
