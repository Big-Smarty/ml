# Chapter 8 research

Route: GPT-5.6 Sol High author with a bounded GPT-5.6 Luna High primary-source research pass. Baydin et al. supports the reverse-mode account; micrograd provided an implementation lineage check; PyTorch documentation verifies that gradients are summed into leaves. The Rust design was independently written and tests the shared-node case directly. Numerical comparison uses the course tolerance and avoids a nondifferentiable point.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (08–14). A finite final tanh output could hide an infinite intermediate. Backward now rejects nonfinite reachable values and overflowing adjoints. Tests cover both finite-output cases and shared accumulation; recursive scalar-graph limits remain explicit.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.


## Section redesign — 2026-09-09

One GPT-6 Astra High owner authored Chapters 07–11 as a connected section. Read original lessons, metadata, research and both project variants before replacement. The existing source verification above grounds the mathematical claims; no new claim of MNIST performance is made. The new lab reuses the reference arithmetic and supplied parsing/checkpoint mechanisms, with an append-only scalar tape replacing Rc/RefCell infrastructure. Core exercises now call learner functions through explicit goal checks; normal tests validate functioning baselines and the separate completed solutions.

The route uses the supplied foundations audit, learning-science synthesis and interactive design research: runnable baseline, collocated arithmetic and code, progressive help, a meaningful implementation and changed-input evidence. These are design inferences, not evidence that this complete course format is experimentally proven superior. Session times are planning estimates, not measured learner times. See `guidance/redesign/section-02.md` for topic/action/check mapping and actual validation.
