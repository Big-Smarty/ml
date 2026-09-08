# Chapter 11 research

Route: GPT-5.6 Sol High author with a bounded GPT-5.6 Luna High primary-source research pass. Polyak supports the momentum lineage; Kingma and Ba supplies Adam’s moments, bias correction, update, and default coefficients; PyTorch’s official tutorial cross-checks the need to save optimizer state for resume. MNIST format facts inherit the verified official source from Chapter 10. Code and generated fixture are original, and the exact-resume test proves that restored moments and step affect the next update.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (08–14). Implemented the documented zero-epoch evaluation path: an existing checkpoint is required and its bytes remain unchanged. Train and held-out image shapes must match, not just their products. Corrected common-offset cross-entropy cancellation in both reference and working starter. Reject overflowing loss/gradients and optimizer moments. Regression tests cover evaluation, state overflow, and two-step starter momentum. Prior lead-measured MNIST smoke results were preserved and were not rerun.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.

## Consistency revision — 2026-09-08

The Chapter 10 data and multiclass-loss contract now carries through directly: stored arrays are `images` and `labels`, the stable primitive is `cross_entropy_from_logits(logits, target)`, and batch loss and gradient both average over examples. Chapter 11 adds a hidden-activation forward cache, a named W1/b1/W2/b2 packing map, and stateful optimizer steps. These are local terminology and interface changes; no new external research, dataset, numerical formula, CLI mode, or checkpoint field was introduced.
