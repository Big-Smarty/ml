# Chapter 9 research

Route: GPT-5.6 Sol High author with a bounded GPT-5.6 Luna High primary-source research pass. MIT matrix-calculus notes verify the product differential; NumPy and PyTorch documentation verify conventional batched matmul dimensions; JAX documentation grounds the vector-Jacobian-product framing. The course fixes weights to [out,in], so formulas and tests were derived for that declared layout rather than copied from a library API.

The consistency pass keeps that algorithm and fixes the teaching interface around it: `Dense { in_features, out_features }`, inputs `[batch,in_features]`, weights `[out_features,in_features]`, outputs `[batch,out_features]`, and `backward(inputs, output_gradients)`. The dense backward pass performs the local vector-Jacobian product only; the upstream scalar loss owns any mean or sum reduction.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (08–14). Verified all nonsquare forward/backward arithmetic, shape errors, and finite differences. Strengthened row-major learner checks with additional rows, columns, and widths. The reference kernel needed no implementation correction.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.
