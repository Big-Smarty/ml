# Chapter 14 research notes

## Route and verification

Authored by Sol High after a bounded GPT-5.6 Luna High researcher browsed canonical and official sources on 2026-09-08. The original nearest-neighbor paper anchors the method; official scikit-learn pages verify current formulations of scaling, neighbor voting, and Gaussian naive Bayes. The implementation and fixture are original.

## Evidence used

- Cover and Hart define nearest-neighbor classification and analyze its asymptotic error: https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf
- The scikit-learn nearest-neighbors guide documents uniform and distance-weighted k-neighbor votes: https://scikit-learn.org/stable/modules/neighbors.html
- `StandardScaler` documents centering by mean and scaling by standard deviation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- The scikit-learn naive Bayes guide states conditional independence, Gaussian per-feature likelihoods, and cautions about probability estimates: https://scikit-learn.org/stable/modules/naive_bayes.html
- UCI lists Iris and its CC BY 4.0 license for the optional extension: https://archive.ics.uci.edu/dataset/53/iris

## Author decisions

The fixed two-feature implementation uses a full distance sort because it reveals the algorithm and is adequate for six rows. `Point.features` is kept separate from its discrete `Point.label`, matching the prior chapter's boundary while making the narrower array shape explicit. `Scaler::fit` estimates training-only state and `transform` reuses it. `Knn::fit` stores the transformed rows and chosen hyperparameters; `GaussianNb::fit` instead estimates priors, means, and population variances. Both `predict` methods return a label. Squared distances and Gaussian log scores remain unnormalized internal comparison values.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (08–14). Gaussian prediction now rejects nonfinite queries and log scores rather than choosing an arbitrary class. Scaling and Gaussian fitting reject overflowing statistics; kNN rejects overflowing squared distances. Added a Gaussian log-density oracle and deterministic vote-tie check. Clarified that a constant training coordinate gives no neighbor discrimination, even when a differing query adds a common distance term.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.
