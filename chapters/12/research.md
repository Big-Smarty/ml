# Chapter 12 research notes

## Route and verification

Authored by Sol High with a bounded GPT-5.6 Luna High research subagent. The researcher browsed primary and authoritative sources on 2026-09-08; the author checked the returned claims against the linked pages and used them to constrain the lesson and implementation. The local fixture, prose, and Rust code are original.

## Evidence used

- Efron’s original bootstrap paper defines bootstrap estimation through resampling from the empirical distribution: https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full
- NIST describes repeated resampling with replacement and recomputation of a statistic: https://www.itl.nist.gov/div898/handbook/eda/section3/bootplot.htm
- The scikit-learn calibration guide defines reliability-diagram coordinates and warns that Brier/log loss combine calibration with discrimination and uncertainty: https://scikit-learn.org/stable/modules/calibration.html
- Kreiss and Lahiri survey dependence-aware bootstrap methods for time series, supporting the warning against ordinary row resampling for dependent observations: https://www.sciencedirect.com/science/article/pii/B9780444538581000016

## Author decisions

The runnable project implements a simple percentile confidence interval because its resampling mechanism is inspectable. The lesson explicitly denies a coverage guarantee for the eight-row fixture and separates bootstrap sampling uncertainty from bias, shift, and dependence. It measures calibration but does not fit a calibrator, avoiding another model and another data partition. The consistency revision maps each binary evaluation row to one frozen Chapter 11 softmax class probability and the corresponding observed class-versus-rest outcome. Accuracy, Brier score, and calibration summaries are recomputed from those held-out rows; none consumes the classifier's recorded training loss.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (08–14). Verified worked confidence-interval, bootstrap, calibration, and Brier arithmetic against Efron and official calibration documentation. Added checks distinguishing all-correct, all-wrong, and mixed bootstrap samples, exact calibration summaries, malformed inputs, and unequal Brier errors. PubMed 24895046 was verified as Austin and Steyerberg’s calibration-curve bootstrap paper.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.


## September 2026 section redesign

The original lesson, metadata, and `projects/ch12` reference were read before replacement; the original source-tagged excerpt and reference project remain available. The new section route connects a disclosed maintenance dataset through uncertainty, preprocessing, classical learning, projection, clustering and a frozen evaluation capstone. Existing source claims above support the mathematical exposition; dataset-specific numbers come from the actual new Rust implementation, not from those papers.

Learner goal: Implement whole-machine percentile bootstrap and counted reliability bins. The chapter has 3 sessions of approximately40 minutes. Topic coverage is recorded verbatim in meta.json. Each session includes implementation and an unfamiliar or controlled transfer, with native hints and explained completed-source checkpoints. Browser calculations are explicitly illustrations and never claim to execute Rust.

Checks: Whole-machine resampling retains paired predictions and duplicated-group invariance; Degenerate predictions and unfamiliar calibration-bin arithmetic; Reject malformed probabilities and inadequate repetitions. Limitations: Only six independent machines; nominal bootstrap coverage is fragile; Group bootstrap preserves each short sequence but does not model shared shocks or regime shift; No recalibrator is fitted; confidence sharpening is a demonstration. Full execution evidence, algorithm scope and prerequisite audit are in guidance/redesign/section-03.md. The original reference may use a simpler fixture or a different algorithm variant; its preserved results are labelled separately from the shared lab.
