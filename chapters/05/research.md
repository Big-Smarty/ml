# Chapter 5 research

Model route: GPT-5.6 Sol High author with bounded GPT-5.6 Luna High primary-source research. Confusion counts and metrics were recomputed in Rust tests.

- scikit-learn's official split API documents reproducible randomized and stratified train/test splitting: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- scikit-learn's official pitfalls guide supports fitting preprocessing only on training data and identifies leakage: https://scikit-learn.org/stable/common_pitfalls.html
- Google's official classification metrics module supports confusion-matrix, accuracy, precision, and recall formulas: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall
- Google's thresholding module supports the explanation that thresholds turn scores into decisions and change error tradeoffs: https://developers.google.com/machine-learning/crash-course/classification/thresholding
- Fawcett (2006) supports ROC interpretation: https://doi.org/10.1016/j.patrec.2005.10.010
- Saito and Rehmsmeier (2015) supports preferring precision-recall analysis for imbalanced data: https://doi.org/10.1371/journal.pone.0118432

The saved probability/target examples, audit design, prose, and code are original.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (01–07). Printed the candidate and baseline confusion counts already promised by the lesson. Threshold selection remains validation-only with unchanged metric conventions.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.
