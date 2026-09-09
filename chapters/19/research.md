# Chapter 19 research notes

## Route and verification

The assigned Sol High author used a bounded GPT-5.6 Luna High researcher to collect primary and first-party evidence. The researcher verified the linked sources on 2026-09-08 and returned cross-validation arithmetic, selection-bias evidence, reproducibility patterns, and debugging cases. The author integrated the material with the course's earlier split and preprocessing conventions and verified the capstone executable.

## Evidence used

- Kohavi, [“A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection”](https://ai.stanford.edu/~ronnyk/accEst.pdf), supports stratified cross-validation as an empirically useful model-selection method on the studied classification datasets.
- Cawley and Talbot, [“On Over-fitting in Model Selection and Subsequent Selection Bias”](https://www.jmlr.org/papers/v11/cawley10a.html), supports the warning that selecting and reporting on the same criterion can produce optimistic conclusions.
- Bergstra and Bengio, [“Random Search for Hyper-Parameter Optimization”](https://jmlr.org/papers/v13/bergstra12a.html), supports fixed-budget random search when only some hyperparameters strongly matter.
- Google, [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml), is first-party guidance for beginning with simple models and a sound pipeline.
- Rust standard library, [slice documentation](https://doc.rust-lang.org/std/primitive.slice.html), verifies the stable sorting API used to make fold assignment deterministic.

## Protocol decisions and arithmetic

The executable freezes 24 development rows, eight test rows, four stratified folds, seed 20260908, binary accuracy, and four logistic configurations. This requires 16 cross-validation fits and one final refit. Scaling is fit separately inside each fold. Accuracy is pooled by row counts, so folds `(8,10)` and `(1,2)` produce `9/12 = 0.75`, not the unweighted 0.65.

The comparison is explicitly majority baseline versus logistic regression. It does not claim broad model-family superiority. Deterministic tie-breaking retains the earliest frozen candidate. The final program prints stable error IDs and describes the perfectly separable fixture as a workflow test, not a deployment estimate.

## Excluded extensions

Nested cross-validation, confidence intervals, grouped and temporal fold code, artifact serialization, and external UCI data were excluded. They are discussed as extensions or limitations rather than simulated. The runnable capstone fully implements the assigned reproducible baseline, search, refit, test, and error-analysis path.


## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 15–19; the researcher supplied checks and source findings, while Astra implemented the corrections.

Validated the complete CV input and unique stable IDs; checked reordered membership. Removed the scaler’s dimensional epsilon cutoff in both reference and starter, tested tiny feature units, and rejected overflowed scaling/training results. Kept train-only scaling and the frozen four-candidate search.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.

## Consistency revision, 2026-09-08

No new online research was needed. The local consistency contract and frozen Chapters 4, 14, and 15 establish `LogisticModel`, `Row.features`, discrete `Row.label`, `Scaler::fit(train_data)`, `Scaler::transform(features)`, `predict(features)`, and `learning_rate`. Chapter 19 keeps the same mathematics while storing fold-fitted scaler statistics beside the learned weights and bias, so `probability` and `predict` accept raw features. Cross-validation pools correct predictions and held-out row counts; it does not take an unweighted mean of fold percentages. The Chapter 5 glossary entry remains the sole owner of `data-leakage`.


## September 2026 section redesign

The original lesson, metadata, and `projects/ch19` reference were read before replacement; the original source-tagged excerpt and reference project remain available. The new section route connects a disclosed maintenance dataset through uncertainty, preprocessing, classical learning, projection, clustering and a frozen evaluation capstone. Existing source claims above support the mathematical exposition; dataset-specific numbers come from the actual new Rust implementation, not from those papers.

Learner goal: Implement frozen group/time cross-validation and six-candidate pooled-cost selection. The chapter has 3 sessions of approximately40 minutes. Topic coverage is recorded verbatim in meta.json. Each session includes implementation and an unfamiliar or controlled transfer, with native hints and explained completed-source checkpoints. Browser calculations are explicitly illustrations and never claim to execute Rust.

Checks: Three group/time folds have48/18 rows and54 distinct validation IDs; All six candidates are evaluated; minimum pooled cost and stable ordering; Reordered source rows return the same sorted partition IDs; checks do not open final outcomes. Limitations: Controlled288-row dataset with only18 final rows and six final machines; The development winner fails to lead on shifted final data; final results do not authorize reselection; Nested CV and causal diagnosis are explained, not implemented; final slices are descriptive. Full execution evidence, algorithm scope and prerequisite audit are in guidance/redesign/section-03.md. The original reference may use a simpler fixture or a different algorithm variant; its preserved results are labelled separately from the shared lab.
