# Chapter 13 research notes

## Route and verification

Authored by Sol High with bounded GPT-5.6 Luna High primary-source research. Sources were browsed on 2026-09-08. Official scikit-learn API and user-guide pages establish the fit/transform behaviors; the UCI page establishes the optional dataset’s provenance and stated license. Code and the default table are original course material.

## Evidence used

- scikit-learn’s common-pitfalls guide states that splitting precedes preprocessing and that transforms must be learned from training data: https://scikit-learn.org/stable/common_pitfalls.html
- `SimpleImputer` documents median imputation and optional missingness indicators: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
- `OneHotEncoder` documents fitted category columns and all-zero output with `handle_unknown="ignore"`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- `GroupKFold` guarantees non-overlapping groups: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
- `TimeSeriesSplit` keeps test indices later than training and documents a gap: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- UCI lists Auto MPG and CC BY 4.0 for the optional extension: https://archive.ics.uci.edu/dataset/9/auto+mpg

## Author decisions

The project exposes fitted state directly: one median and a sorted vocabulary. Unknown categories map to all zeros, matching a documented common policy. A test puts an extreme value and novel category only in the held-out group, so leakage changes observable expected values. General CSV parsing and a combined constraint solver were excluded.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (08–14). The even median could overflow from finite same-sign values. Native f64::midpoint preserves finite midpoints; tests now cover extreme medians, missingness indicators, all-missing fitting, and nonfinite transforms. The API is stable since Rust 1.85: https://doc.rust-lang.org/std/primitive.f64.html#method.midpoint . Leakage and split invariants remain unchanged.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.


## September 2026 section redesign

The original lesson, metadata, and `projects/ch13` reference were read before replacement; the original source-tagged excerpt and reference project remain available. The new section route connects a disclosed maintenance dataset through uncertainty, preprocessing, classical learning, projection, clustering and a frozen evaluation capstone. Existing source claims above support the mathematical exposition; dataset-specific numbers come from the actual new Rust implementation, not from those papers.

Learner goal: Implement training-fitted median/scaling/vocabulary and four legal split policies. The chapter has 3 sessions of approximately40 minutes. Topic coverage is recorded verbatim in meta.json. Each session includes implementation and an unfamiliar or controlled transfer, with native hints and explained completed-source checkpoints. Browser calculations are explicitly illustrations and never claim to execute Rust.

Checks: Median4 and imputed mean27.5 for changed missing-value fixture; Training fitted state is invariant to held-only extreme values and categories; Row/group/time/strict membership, chronology, unknown encoding and constant scales. Limitations: Median/indicator and one-hot choices are simple modeling assumptions; All-missing training numeric columns are rejected; unknown categories have an explicit bit; Splits answer different population questions and are not a model-selection contest. Full execution evidence, algorithm scope and prerequisite audit are in guidance/redesign/section-03.md. The original reference may use a simpler fixture or a different algorithm variant; its preserved results are labelled separately from the shared lab.
