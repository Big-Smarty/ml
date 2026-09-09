# Chapter 15 research notes

## Route and verification

Authored by Sol High with a bounded GPT-5.6 Luna High researcher assigned to primary-source browsing. Sources were reviewed on 2026-09-08. Original papers ground random forests and gradient boosting; official scikit-learn guides verify CART terminology, regularization controls, and ensemble behavior. The fixture and Rust code are original.

## Evidence used

- The scikit-learn tree guide describes optimized CART, recursive partitions, piecewise-constant predictions, and depth/leaf complexity controls: https://scikit-learn.org/stable/modules/tree.html
- Breiman’s original Random Forests paper relates ensemble error to tree strength and correlation and defines randomized tree ensembles: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
- Friedman formulates gradient boosting as stagewise additive function approximation in function space: https://doi.org/10.1214/aos/1013203451
- The scikit-learn ensemble guide documents bootstrap aggregation, per-node random feature subsets, out-of-bag estimation, and learning-rate/estimator tradeoffs: https://scikit-learn.org/stable/modules/ensemble.html

## Author decisions

The baseline tree searches both features deterministically. The forest separately bootstraps rows for each tree and samples a fresh candidate feature inside every recursive node, which preserves the defining random-forest behavior. Boosting uses squared-error regression so negative gradients reduce to residuals that learners can calculate by hand. Out-of-bag bookkeeping and classification loss remain stated limitations.


## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 15–19; the researcher supplied checks and source findings, while Astra implemented the corrections.

Corrected a second worked split that changed the parent class counts. Strengthened the tree/forest test to classify both sides and checked the mixed-node impurity.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.


## September 2026 section redesign

The original lesson, metadata, and `projects/ch15` reference were read before replacement; the original source-tagged excerpt and reference project remain available. The new section route connects a disclosed maintenance dataset through uncertainty, preprocessing, classical learning, projection, clustering and a frozen evaluation capstone. Existing source claims above support the mathematical exposition; dataset-specific numbers come from the actual new Rust implementation, not from those papers.

Learner goal: Implement recursive CART, bagging, random feature forests and residual boosting. The chapter has 4 sessions of approximately40 minutes. Topic coverage is recorded verbatim in meta.json. Each session includes implementation and an unfamiliar or controlled transfer, with native hints and explained completed-source checkpoints. Browser calculations are explicitly illustrations and never claim to execute Rust.

Checks: Recursive XOR partition distinguishes a median stump; Depth/node changes, genuinely resampled bagging and random-feature forest; Squared-residual boosting lowers held synthetic regression error. Limitations: Small-data CART and fixed iteration counts are inspectable rather than optimized; Boosting minimizes squared residuals and clips only reported probabilities; it is not logistic gradient boosting; Trees/forests use fixed declared seeds and small ensembles; validation can worsen. Full execution evidence, algorithm scope and prerequisite audit are in guidance/redesign/section-03.md. The original reference may use a simpler fixture or a different algorithm variant; its preserved results are labelled separately from the shared lab.
