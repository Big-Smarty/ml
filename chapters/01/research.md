# Chapter 1 research

Authored directly by the lead, following the six Luna High research reports summarized in guidance/RESEARCH.md. Numerical examples were checked by executing the Rust reference. The synthetic line, text and visualization are original. Google ML Crash Course supports the linear-regression framing; Rust documentation supports implementation and tooling.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (01–07). Reviewed the full quality-reference lesson, numerical-gradient implementation, browser demo source, starter, and Rustlings pair. Preserved initial MSE 9 and first update (0.8, 0.2) with loss 3.52. The starter now also tests different weight, bias, and input values. Google’s own regression documentation and the existing language/tool documentation remain the directly relevant sources.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.

## Consistency revision — 2026-09-08

No new online research was needed. The revision follows the local `guidance/CONSISTENCY.md`, `guidance/AUTHORING.md`, and `guidance/NUMERICS.md` contracts. It makes the reference, starter, lesson snippets, and Rustlings pair use the same `Neuron` method vocabulary and keeps the existing synthetic fixture and numerical results. The learner's completed prediction and 1,000-step starter experiment were preserved while their free functions were mapped to the canonical methods. An asymmetric regression fixture now detects sequential weight/bias updates that the symmetric course data can mask.
