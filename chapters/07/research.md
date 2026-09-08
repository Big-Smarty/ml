# Chapter 7 research

Route: GPT-5.6 Sol High author with a bounded GPT-5.6 Luna High primary-source research pass. The researcher verified the XOR linear-separability argument in the Stanford PDP Handbook, the historical perceptron source, and the 1986 back-propagation paper. The implemented stable binary cross-entropy is derived directly from logits; both its output and hidden-layer gradients were checked against central differences. All XOR data, prose, and Rust code are course-authored.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (01–07). Separated one-example loss ℓ from mean loss L in the backward formulas and introduced tanh before the transfer exercise uses it. Added central differences against the actual update for all nine parameters; retained both original layer checks. Strengthened sigmoid-derivative checks away from 0.5.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.

## Consistency revision — 2026-09-08

No new online research was needed. The revision preserves the verified XOR mathematics and stable logit loss while making the teaching interface match the course contract: data is explicit, `forward` names reusable intermediates, `probability` and class-valued `predict` name task outputs, `gradient` returns a separate mean-reduced `Gradient`, and `step` applies it with `learning_rate`. The XOR truth table and deterministic parameter initialization are course-authored fixtures. The lesson's tagged Rust excerpt is copied exactly from `projects/ch07/src/main.rs`.
