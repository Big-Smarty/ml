# Chapter 7 research

Route: GPT-5.6 Sol High author with a bounded GPT-5.6 Luna High primary-source research pass. The researcher verified the XOR linear-separability argument in the Stanford PDP Handbook, the historical perceptron source, and the 1986 back-propagation paper. The implemented stable binary cross-entropy is derived directly from logits; both its output and hidden-layer gradients were checked against central differences. All XOR data, prose, and Rust code are course-authored.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (01–07). Separated one-example loss ℓ from mean loss L in the backward formulas and introduced tanh before the transfer exercise uses it. Added central differences against the actual update for all nine parameters; retained both original layer checks. Strengthened sigmoid-slope checks away from 0.5.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.
