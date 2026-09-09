# Chapter 9 research

Route: GPT-5.6 Sol High author with a bounded GPT-5.6 Luna High primary-source research pass. MIT matrix-calculus notes verify the product differential; NumPy and PyTorch documentation verify conventional batched matmul dimensions; JAX documentation grounds the vector-Jacobian-product framing. The course fixes weights to [out,in], so formulas and tests were derived for that declared layout rather than copied from a library API.

The consistency pass keeps that algorithm and fixes the teaching interface around it: `Dense { in_features, out_features }`, inputs `[batch,in_features]`, weights `[out_features,in_features]`, outputs `[batch,out_features]`, and `backward(inputs, output_gradients)`. The dense backward pass performs the local vector-Jacobian product only; the upstream scalar loss owns any mean or sum reduction.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (08–14). Verified all nonsquare forward/backward arithmetic, shape errors, and finite differences. Strengthened row-major learner checks with additional rows, columns, and widths. The reference kernel needed no implementation correction.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.


## Section redesign — 2026-09-09

One GPT-6 Astra High owner authored Chapters 07–11 as a connected section. Read original lessons, metadata, research and both project variants before replacement. The existing source verification above grounds the mathematical claims; no new claim of MNIST performance is made. The new lab reuses the reference arithmetic and supplied parsing/checkpoint mechanisms, with an append-only scalar tape replacing Rc/RefCell infrastructure. Core exercises now call learner functions through explicit goal checks; normal tests validate functioning baselines and the separate completed solutions.

The route uses the supplied foundations audit, learning-science synthesis and interactive design research: runnable baseline, collocated arithmetic and code, progressive help, a meaningful implementation and changed-input evidence. These are design inferences, not evidence that this complete course format is experimentally proven superior. Session times are planning estimates, not measured learner times. See `guidance/redesign/section-02.md` for topic/action/check mapping and actual validation.
