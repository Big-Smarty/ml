# Chapter 4 research

Model route: GPT-5.6 Sol High author with bounded GPT-5.6 Luna High primary-source research. Stable-loss behavior was verified at logits ±1000 in the Rust project.

- Berkson's 1944 paper provides historical grounding for applying the logistic function: https://doi.org/10.1080/01621459.1944.10500699
- Penn State STAT 504 gives the Bernoulli likelihood and logistic-regression derivation used to check the exposition: https://online.stat.psu.edu/stat504/Lesson06
- SciPy's official `expit` documentation confirms the sigmoid definition: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html
- PyTorch's official `BCEWithLogitsLoss` documentation supports combining logits with binary cross-entropy for numerical stability: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- scikit-learn's official logistic-regression guide supports probability and classification framing: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

The two-class points and implementation are original course fixtures.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (01–07). Broadened the shared logit glossary entry to distinguish binary log-odds from multiclass relative scores. Strengthened sigmoid and stable-BCE learner checks with known probabilities, ln 2, and confident wrong labels.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.
