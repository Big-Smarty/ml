# Section 03 — Beyond neural networks (Chapters 12–19)

Complete and frozen, 2026-09-08. Reviewed against base commit `43a16cc2c0b09a2bc4c75f79869cc2b9ab4fca5c`.

## Coverage

Read all eight `chapters/NN/lesson.html` and `meta.json` files, their course assignments, all eight `projects/chNN/src/main.rs` references, all eight starter sources, both reference/starter Cargo manifests, and all sixteen matching `exercises/cpu/{exercises,solutions}/chNN_01.rs` files. Read the shared consistency, numerics, and authoring contracts and applicable Rust review skills. Checked Chapter 11's probability/checkpoint interface and Chapter 20's continuity boundary.

- 12: paired bootstrap sampling, percentile indexing and coverage cautions, binary calibration and Brier reductions.
- 13: split-before-fit, median/missingness, category vocabulary, grouped and temporal boundaries.
- 14: two-feature transition, scaling, distance ranking/ties, Gaussian log densities and class decisions.
- 15: weighted Gini, recursive splits, bootstrap/per-node feature randomness, residual sign and boosting SSE/MSE relationship.
- 16: signed labels, functional/geometric margins, hinge-plus-penalty subgradient update, kernel perceptron versus kernel SVM.
- 17: sample covariance, eigenvectors, power iteration, reconstruction reduction, conditioning and documented solver limits.
- 18: final k-means assignments/inertia, EM responsibilities and updates, log-space normalization, density-based anomaly interpretation.
- 19: fold construction, stable row identity, fold-only preprocessing, logistic gradient reduction, pooled accuracy, fixed search/ties, untouched test boundary.

## Material findings and fixes

1. **Chapter 19 could construct empty folds for valid small classes.** With two rows per class and three requested folds, restarting the round-robin counter for each class put all four rows into folds 0 and 1. The resulting split failed the downstream CV check despite the lesson claiming nonempty folds by construction. The new `small_classes_still_fill_and_balance_all_folds` test failed on the original implementation at the nonempty-fold assertion. Continue round-robin placement across the class-sorted sequence instead. Each class occupies a consecutive segment of the sequence, so its per-fold counts differ by at most one; total fold sizes also differ by at most one, and `k <= rows.len()` ensures every fold is populated. The test checks both balances and successful downstream CV. Updated the lesson's construction explanation. No signature changes. The default 24-row/four-fold stdout is byte-identical before and after, including all fold IDs, candidate scores, selected configuration, and test results.

2. **Chapter 18's unequal-spread worked answer implied an unconditional density ordering.** Greater spread does not automatically imply greater density at a specified displacement: Gaussian normalization and the distance exponent compete. Replaced the unsupported general prediction with equal-weight components having means `[-2,-2]` and `[2,2]` and variances `[1,1]` and `[0.04,0.04]`. Points `[-3,-2]` and `[3,2]` both have nearest-center squared distance 1, while full-mixture negative log densities are `3.0310242469692907` and `11.812135003686002`. The lesson gives rounded values and explains why other distances, spreads, weights, and component contributions can change the ordering. No algorithm change.

## Validation

For each `NN = 12, 13, 14, 15, 16, 17, 18, 19`, ran:

```text
cargo fmt --manifest-path projects/chNN/Cargo.toml --check
cargo clippy --offline --manifest-path projects/chNN/Cargo.toml --all-targets -- -D warnings
cargo test --offline --manifest-path projects/chNN/Cargo.toml
cargo run --offline --quiet --manifest-path projects/chNN/Cargo.toml
cargo fmt --manifest-path projects/chNN/starter/Cargo.toml --check
cargo clippy --manifest-path projects/chNN/starter/Cargo.toml --offline --all-targets -- -D warnings
cargo run --manifest-path projects/chNN/starter/Cargo.toml --offline --quiet
cargo test --manifest-path projects/chNN/starter/Cargo.toml --offline
rustc --edition=2021 --test exercises/cpu/exercises/chNN_01.rs -o /tmp/ml-section03-exercises-chNN
/tmp/ml-section03-exercises-chNN
rustc --edition=2021 --test exercises/cpu/solutions/chNN_01.rs -o /tmp/ml-section03-solutions-chNN
/tmp/ml-section03-solutions-chNN
```

All reference format, strict Clippy, tests, and bounded default demos passed. Reference test counts: 2, 2, 2, 1, 4, 6, 5, 5 (27 total). All starter format/Clippy checks and default launches passed. All eight starter tests and all eight exercise tests compiled and failed specifically at their intended `todo!`; all eight solution test binaries passed. No learner TODO was altered.

A scoped HTML parser check verified all 34 tagged source excerpts in Chapters 12–19 against their actual source, ignoring whitespace. Independently evaluated the Chapter 18 full-mixture arithmetic above. Compared Chapter 19 before/after demo output with `diff -u`: no differences. Temporary command logs are in `/tmp/ml-section03-validation.json`; no existing validation evidence was modified.

## Limits and cross-section status

No remaining material cross-section issue identified. No shared interfaces, shared files, data/checkpoint artifacts, existing review logs, or learner work changed. No new dependencies, network datasets, extended training, GPU checks, global builds, or global verifiers were needed. Validation covers the bounded local fixtures and documented teaching interfaces; it does not establish real-world model quality or production numerical robustness beyond the lessons' stated limits. Root retains integrated site/build and publication checks.
