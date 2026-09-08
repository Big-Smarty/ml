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
