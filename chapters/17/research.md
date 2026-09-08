# Chapter 17 research notes

## Route and verification

The assigned Sol High author used a bounded GPT-5.6 Luna High research subtask focused on primary sources. The researcher verified source titles and URLs on 2026-09-08 and supplied derivations, conditioning hazards, and transfer checks. The author integrated those findings into an original two-dimensional implementation and verified the arithmetic locally.

## Evidence used

- Pearson, [“On Lines and Planes of Closest Fit to Systems of Points in Space”](https://doi.org/10.1080/14786440109462720), supplies the geometric closest-fit origin of PCA.
- Hotelling, [“Analysis of a Complex of Statistical Variables into Principal Components”](https://doi.org/10.1037/h0071325), supplies the statistical principal-components formulation.
- MIT OpenCourseWare, [Lecture 17: PCA](https://ocw.mit.edu/courses/9-40-introduction-to-neural-computation-spring-2018/resources/mit9_40s18_lec17/), was used to cross-check covariance, projection, reconstruction, and explained-variance presentation.
- University of Illinois CS 357, [Eigenvalues and Eigenvectors](https://courses.physics.illinois.edu/cs357/sp2020/notes/ref-12-eigen.html), supports the power-iteration recurrence and convergence conditions.
- UCI, [Wine dataset](https://archive.ics.uci.edu/dataset/109/wine), is the optional external extension. Its dataset page lists CC BY 4.0; no Wine data is bundled or downloaded by default.

## Arithmetic and implementation decisions

The chapter uses sample covariance with denominator `n - 1`. For the hand matrix `[[2,1],[1,2]]`, eigenpairs are 3 with `[1,1]/sqrt(2)` and 1 with `[1,-1]/sqrt(2)`, so one component retains 75%. Starting power iteration at `[1,0]` yields the stated Rayleigh quotients 2.8, approximately 2.9756, and approximately 2.9973.

A single fixed start can be orthogonal to the dominant eigenvector. The two-dimensional reference therefore runs two independent starts and selects the larger Rayleigh quotient. This is intentionally smaller than a general eigensolver but is accompanied by unit-norm, eigenpair-residual, translation-invariance, and exact rank-one reconstruction checks.

The condition-number discussion distinguishes a nearly rank-one covariance from a small leading eigengap. Explicit covariance squares the centered data matrix's condition number, motivating the stated production preference for SVD.

## Excluded extensions

Generic dimensions, top-k deflation, randomized SVD, streaming covariance, and file parsing were excluded. The assigned project is the full promised two-feature compressor and exposes the numerical path without a tensor abstraction.



## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 15–19; the researcher supplied checks and source findings, while Astra implemented the corrections.

Removed dimension-dependent machine-epsilon cutoffs from direction normalization and conditioning; used hypot and a scaled two-by-two eigenvalue ratio. Constant covariance now has an explicit arbitrary-axis and zero-reported-fraction convention. Added unit-scale invariance and constant-data tests.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.
