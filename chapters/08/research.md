# Chapter 8 research

Route: GPT-5.6 Sol High author with a bounded GPT-5.6 Luna High primary-source research pass. Baydin et al. supports the reverse-mode account; micrograd provided an implementation lineage check; PyTorch documentation verifies that gradients are summed into leaves. The Rust design was independently written and tests the shared-node case directly. Numerical comparison uses the course tolerance and avoids a nondifferentiable point.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (08–14). A finite final tanh output could hide an infinite intermediate. Backward now rejects nonfinite reachable values and overflowing adjoints. Tests cover both finite-output cases and shared accumulation; recursive scalar-graph limits remain explicit.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.
