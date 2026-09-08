# Chapter 16 research notes

## Route and verification

The chapter was authored by the assigned Sol High author after bounded primary-source research by a GPT-5.6 Luna High researcher. The researcher verified the source titles and landing pages on 2026-09-08 and returned equations, implementation hazards, and exercise candidates. The author reconciled those findings with the local authoring and numerical contracts and ran the independent Rust project. This file records evidence; it is not lecture prose.

## Evidence used

- Cortes and Vapnik, [“Support-Vector Networks”](https://doi.org/10.1007/BF00994018), is the original 1995 soft-margin support-vector network paper. It supports the margin/slack formulation and kernel decision function.
- Boser, Guyon, and Vapnik, [“A Training Algorithm for Optimal Margin Classifiers”](https://doi.org/10.1145/130385.130401), supports replacing inner products with kernel evaluations in the optimal-margin formulation.
- Platt, [“Sequential Minimal Optimization”](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/), is the primary source for a practical dual kernel-SVM optimizer. It is cited to define what this chapter deliberately does not implement.
- Stanford Statistics 202, [Support Vector Machines](https://web.stanford.edu/class/stats202/slides/Support-vector-machines.html), was used as primary university teaching material to cross-check margin distance, hinge loss, and RBF notation.

## Decisions and checks

The executable uses the averaged primal objective `lambda/2 * ||w||^2 + mean hinge` and states that its constants differ from the common `C * sum hinge` convention. The nonlinear path is explicitly a kernel perceptron. Calling it a kernel SVM would overstate the implementation because it does not solve the constrained maximum-margin dual.

The worked RBF values were independently recalculated: with gamma 0.5, squared distances one and four yield `exp(-0.5) = 0.6065` and `exp(-2) = 0.1353`. Tests establish that the zero linear model has average hinge one, the linear fixture is separated, the same linear path fails on XOR, and the RBF kernel classifier fits XOR. Training fixtures are never described as held-out evidence.

## Excluded extensions

SMO, exact dual coefficients, calibrated probabilities, Gram-matrix caching, and external datasets were excluded. They add machinery without strengthening the assigned contrast between a linear hinge-loss SVM and a nonlinear kernel classifier.



## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 15–19; the researcher supplied checks and source findings, while Astra implemented the corrections.

Corrected the margin glossary: absolute score divided by weight norm is distance. Added finite parameter/objective checks and a regression rejecting finite but divergent training settings. The nonlinear model remains explicitly a kernel perceptron, not a kernel SVM.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.
