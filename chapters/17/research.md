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

## Consistency revision

The fitted interface now names its stages explicitly: `Pca1::fit(rows)` estimates and stores the two-feature mean, sample covariance, leading component, and eigenvalue; `Pca1::transform(features)` applies those fitted quantities without refitting and returns a geometric projection score. That projection score is distinct from Chapter 16's classifier score. The Rustlings `projection_score(centered, component)` function isolates only the final dot product, while the starter's `power_step` isolates one iteration from `power_iteration_from`.

The former `reconstruction_mse` operation was renamed `mean_squared_reconstruction_norm` because its unchanged calculation is `sum_i ||features_i - reconstruction_i||_2^2 / n`. It averages squared Euclidean norms over rows, not squared coordinate residuals over `2n` coordinates. This preserves the numerical result and displayed output while distinguishing the reduction from Chapter 22's coordinate MSE.

## Excluded extensions

Generic dimensions, top-k deflation, randomized SVD, streaming covariance, and file parsing were excluded. The assigned project is the full promised two-feature compressor and exposes the numerical path without a tensor abstraction.



## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 15–19; the researcher supplied checks and source findings, while Astra implemented the corrections.

Removed dimension-dependent machine-epsilon cutoffs from direction normalization and conditioning; used hypot and a scaled two-by-two eigenvalue ratio. Constant covariance now has an explicit arbitrary-axis and zero-reported-fraction convention. Added unit-scale invariance and constant-data tests.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.


## September 2026 section redesign

The original lesson, metadata, and `projects/ch17` reference were read before replacement; the original source-tagged excerpt and reference project remain available. The new section route connects a disclosed maintenance dataset through uncertainty, preprocessing, classical learning, projection, clustering and a frozen evaluation capstone. Existing source claims above support the mathematical exposition; dataset-specific numbers come from the actual new Rust implementation, not from those papers.

Learner goal: Implement full covariance, orthogonal power components and reconstruction diagnostics. The chapter has 4 sessions of approximately40 minutes. Topic coverage is recorded verbatim in meta.json. Each session includes implementation and an unfamiliar or controlled transfer, with native hints and explained completed-source checkpoints. Browser calculations are explicitly illustrations and never claim to execute Rust.

Checks: Known covariance eigenvalue3, norm and eigenpair residual tolerances; Two components are orthogonal; translated and rank-one3D reconstruction; Constant observations, feature width and finite arithmetic. Limitations: Covariance formation squares the data matrix condition number; Fixed power iterations may converge slowly near tied eigenvalues; Sensor-only PCA discards categorical/missingness bits and can lose predictive directions. Full execution evidence, algorithm scope and prerequisite audit are in guidance/redesign/section-03.md. The original reference may use a simpler fixture or a different algorithm variant; its preserved results are labelled separately from the shared lab.
