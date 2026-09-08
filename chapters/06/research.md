# Chapter 6 research

Model route: GPT-5.6 Sol High author with bounded GPT-5.6 Luna High primary-source research. Learning curves and regularization effects were measured by the deterministic Rust program.

- Robbins and Monro (1951) is the foundational stochastic-approximation source: https://doi.org/10.1214/aoms/1177729586
- Bottou's *Large-Scale Machine Learning with Stochastic Gradient Descent* supports the stochastic/minibatch optimization discussion: https://leon.bottou.org/papers/bottou-2010
- Hoerl and Kennard (1970) is the original ridge-regression source for L2 coefficient shrinkage: https://doi.org/10.1080/00401706.1970.10488634
- Srivastava et al. (2014) is included to distinguish dropout, a later neural-network regularizer, from the L2 method implemented here: https://jmlr.org/papers/v15/srivastava14a.html

The noisy regression fixture is course-authored. It demonstrates behavior but is not a benchmark or a claim about production data.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (01–07). Moved rate and L2 validation to the training boundary so zero epochs cannot bypass it. Added short-batch, unpenalized-bias, and invalid-zero-epoch checks. Strengthened L2 learner checks for zero regularization and negative weights.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.
