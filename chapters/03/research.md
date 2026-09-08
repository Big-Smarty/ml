# Chapter 3 research

Model route: GPT-5.6 Sol High author with bounded GPT-5.6 Luna High primary-source research. Numerical claims were checked with the chapter's Rust tests and demonstration.

- MIT OpenCourseWare 18.06 supports the vector and dot-product foundations: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- Google ML Crash Course's official linear-regression module supports the multi-feature affine model and gradient-descent framing: https://developers.google.com/machine-learning/crash-course/linear-regression
- scikit-learn's official `LinearRegression` reference supports ordinary least squares and coefficient/intercept conventions: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- scikit-learn's official `StandardScaler` reference supports centering, variance scaling, and storing training-set statistics: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

The synthetic housing-like measurements are invented teaching data, not market evidence. All prose and implementation are original.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (01–07). Clarified that interaction coefficients [2,3,4] describe raw features; standardized coefficients and the intercept transform with training scales and means. Added an independent finite-difference check for all three weights and bias.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.
