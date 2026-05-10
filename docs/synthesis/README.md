# LARQL Synthesis Documents

A focused set of documents intended to build a coherent mental model
of the LARQL technique — what it does, the math underneath it, the
shape of the current Rust implementation, the realistic ecosystem
implications, and answers to a few specific questions about
efficiency, simplification, and a Java toy port.

These are synthesis documents written from a deep dive across the
existing repository documentation and source. Where they conflict
with the originals (`docs/format.md`, `docs/training-free-insert.md`,
`docs/findings.md`, `docs/walk-boundary-sweep.md`,
`crates/*/src/*`), the originals are authoritative.

## Documents

### Foundation set

| # | Document | What you'll learn |
|---|----------|-------------------|
| 1 | [01-explainer.md](01-explainer.md) | High-level explainer: how the technique works, the capabilities it provides |
| 2 | [02-mathematical-foundation.md](02-mathematical-foundation.md) | Rigorous mathematical foundation: equations, the walk approximation, the constellation insert, correctness and limits |
| 3 | [03-software-architecture.md](03-software-architecture.md) | Architecture of the current Rust implementation: crate map, key data structures, extension points, what exists vs. what's planned |
| 4 | [04-executive-summary.md](04-executive-summary.md) | One-to-two-page executive summary: what it is, why it matters, ecosystem implications, honest limitations |
| 5 | [05-questions.md](05-questions.md) | Direct answers to three questions: is the graph view "the real" computation? can it be boiled down? would a Java toy on GPT-2 / nanochat work? |

### Applied — MLX `lm_head` acceleration

| # | Document | What you'll learn |
|---|----------|-------------------|
| 6 | [06-mlx-plan-review.md](06-mlx-plan-review.md) | Critical review of an external plan to add a random-projection `lm_head` accelerator to mlx-lm for the Qwen 3.5 family. Identifies engineering, algorithm, and validation issues. |
| 7 | [07-mlx-plan-revised.md](07-mlx-plan-revised.md) | Revised version of the same sprint plan that fixes the issues identified in 06. PCA-primary instead of random-projection; first-class tied-embedding support; explicit MTP-interaction phase; tightened benchmark methodology. |

## Reading order

- For an executive who needs to make a 5-minute decision: read **04**.
- For an engineer who needs to *use* LARQL and understand it: read
  **01** then **03**.
- For a researcher reproducing or extending the technique: read
  **02** then **03**.
- For a port to another language or environment: read **02**, then
  the relevant section of **05**.
