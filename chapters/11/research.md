# Chapter 11 research

Route: GPT-5.6 Sol High author with a bounded GPT-5.6 Luna High primary-source research pass. Polyak supports the momentum lineage; Kingma and Ba supplies Adam’s moments, bias correction, update, and default coefficients; PyTorch’s official tutorial cross-checks the need to save optimizer state for resume. MNIST format facts inherit the verified official source from Chapter 10. Code and generated fixture are original, and the exact-resume test proves that restored moments and step affect the next update.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (08–14). Implemented the documented zero-epoch evaluation path: an existing checkpoint is required and its bytes remain unchanged. Train and held-out image shapes must match, not just their products. Corrected common-offset cross-entropy cancellation in both reference and working starter. Reject overflowing loss/gradients and optimizer moments. Regression tests cover evaluation, state overflow, and two-step starter momentum. Prior lead-measured MNIST smoke results were preserved and were not rerun.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.
