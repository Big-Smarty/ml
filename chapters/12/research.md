# Chapter 12 research notes

## Route and verification

Authored by Sol High with a bounded GPT-5.6 Luna High research subagent. The researcher browsed primary and authoritative sources on 2026-09-08; the author checked the returned claims against the linked pages and used them to constrain the lesson and implementation. The local fixture, prose, and Rust code are original.

## Evidence used

- Efron’s original bootstrap paper defines bootstrap estimation through resampling from the empirical distribution: https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full
- NIST describes repeated resampling with replacement and recomputation of a statistic: https://www.itl.nist.gov/div898/handbook/eda/section3/bootplot.htm
- The scikit-learn calibration guide defines reliability-diagram coordinates and warns that Brier/log loss combine calibration with discrimination and uncertainty: https://scikit-learn.org/stable/modules/calibration.html
- Kreiss and Lahiri survey dependence-aware bootstrap methods for time series, supporting the warning against ordinary row resampling for dependent observations: https://www.sciencedirect.com/science/article/pii/B9780444538581000016

## Author decisions

The runnable project implements a simple percentile interval because its resampling mechanism is inspectable. The lesson explicitly denies a coverage guarantee for the eight-row fixture and separates bootstrap sampling uncertainty from bias, shift, and dependence. It measures calibration but does not fit a calibrator, avoiding another model and another data partition.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (08–14). Verified worked confidence-interval, bootstrap, calibration, and Brier arithmetic against Efron and official calibration documentation. Added checks distinguishing all-correct, all-wrong, and mixed bootstrap samples, exact calibration summaries, malformed inputs, and unequal Brier errors. PubMed 24895046 was verified as Austin and Steyerberg’s calibration-curve bootstrap paper.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.
