# Section 01 design and validation record

Owner: GPT-6 Astra High, chapters01–06 as one bundle. Project: sensor calibration → multiple measurements → fault probabilities → grouped evaluation → reliable optimization. All original reference projects are preserved. Active practice is the independent `labs/s01-foundations` Cargo package; no old exercise workflow is used by these lessons.

## Chapter design matrix

| Chapter and original topics | Observable objectives and explanation steps | Meaningful learner change | Numerical check and independent transfer | Sessions |
|---|---|---|---|---|
| 01: Prediction; weights and bias; squared loss; numerical slopes; training and inference | `sensor` defines baseline outputs in plain language and displays every candidate score; `line` traces multiply/add; `errors` computes all residual squares; `slopes` computes four probes; `train` performs simultaneous updates; `unseen` separates inference and extrapolation | Retain/test `candidate_search`, construct a six-rule grid, then implement complete numerical-gradient training without overwriting the comparison | MSE<1e-8 and unfamiliar prediction error<1e-4 on two lines; asymmetric first update[1.3,0.8]. Runnable rate/step/alternate-line/outlier variations. Explain why an outlier defeats exact fit | 2:35,37min |
| 02: Derivatives; gradients; analytical updates; finite-difference checks | `repeated-work` counts actual gradient arithmetic; `local-change` derives2e from an expanded square; `derive` averages both components; `gradient-lab` compares probe distances; `same-state` exposes sequential updates; `cost-and-transfer` verifies unchanged learning | Replace the numerical gradient with an analytical one-pass loop, truthful row counter and local diagnostic cases | Three nonzero/asymmetric settings agree within atol1e-6+rtol1e-4; row visitsN instead of4N; unseen x2.5→6 within0.001. Duplicate-data invariant and independently changed line | 2:36,36min |
| 03: Vectors; dot products; linear regression; feature scaling | `missing-measurements` shows a scalar contradiction; `vectors` gives ordered products/shapes; `vector-gradient` derives every component; `scaling` traces mean/std and unstable raw units; `coordinate-experiment` changes units consistently; `interactions` constructs a product feature | Extend the temperature-only gradient to all three feature coordinates and bias; add contribution/unit-change diagnostics | All four components agree with finite differences; raw[.5,500,−.5]→4 within1e-5; constant columns rejected and scaled sums centered. Runnable load-unit rescaling plus independent interaction feature | 2:37,38min |
| 04: Probability; sigmoid; likelihood; stable binary cross-entropy | `fault-question` runs a smoothed prior; `scores` distinguishes three outputs; `likelihood` derives observed probability and mean negative log; `stable-loss` handles extreme logits; `train-classifier` derivesp−y and the update; `decisions-and-transfer` changes threshold/labels | Construct a complete candidate score/probability/loss evaluator, then replace the prior-only function with the complete logistic trainer | BCE<.04; unfamiliar decisions and reversed-label fit; first positive[2,1] update[.1,.05,.05]; all three BCE derivatives checked; confident wrong±1000 logits yield1000. Runnable threshold/reversed-label/contradictory-duplicate controls | 2:37,38min |
| 05: Splits; leakage; metrics; thresholds; baselines | `new-machines` states deployment question; `split-roles` assigns decisions; `group-boundary` groups identity; `leakage` audits field timing and fitted preprocessing; `counts` computes denominators; `choose-policy` selects validation cost; `audit-report` reports residuals | Implement complete machine-level splitting, then minimum-validation-cost threshold selection with deterministic ties; write an audit report | Every row exactly once; pairwise group disjointness; reversed row order preserves assignments; multiple unfamiliar score sets and tie cases. Runnable third-visit/reordering, extra threshold and changed miss-cost controls | 2:36,40min |
| 06: Learning rates; minibatches; regularization; generalization | `curves` defines axes; `rates` computes overshoot; `minibatches` explains actual divisors/epochs; `regularization` derives2λw; `generalization` separates objectives and transfer; `early-stopping` saves the best model; `section-result` compares evidence | Extend full-batch SGD to shuffled uneven minibatches, L2 and saved-best early stopping with persistent RNG/update count | 240 steps for80 epochs of12rows with batch5; SGD validation MSE<.2; L2 shrinkage and finite-difference penalty; an unfamiliar short-batch example checks unpenalized bias; multiple seeds; saved model matches minimum observed validation. Runnable rate/batch/L2/seed/patience/row-count/rotated-label/saturation controls | 2:38,45min |

Every session includes implementation and an independent variation or explanation. Exact metadata topic keys match `course.json`; values name actual ordered DOM steps. There are38 steps and12 sessions in this section. Session times are planning estimates, not enforced pacing. Free-text explanations are self-assessed against worked answers; numerical checks do not grade them.

## Prerequisite audit

The learner needs everyday Rust arrays, loops, functions, borrowing and Result, not prior calculus or ML.01 introduces the operational meaning of parameters/MSE before naming them and defines `Model=[f64;2]` beside the first source excerpt.02 derives the square derivative and chain-rule multiplication from actual changes.03 introduces vector order, scalar reduction, parameter/gradient dimensions, mean and square root before scaling.04 introduces scores and odds before likelihood; sigmoid stability and the combined logit loss precede derivatives.05 recalls probabilities and decisions before ratios and split protocols.06 recalls gradients/evaluation, defines epoch versus step, and works through L2, unequal batch divisors and saved-model selection before code.

Short exact snippets use `data-source` and link to new lab source. Meaningful old section anchors in01–06 are preserved inside their new relevant learning steps;01's old `neuron-lab` anchor is also retained. Point-of-use glossary links are present after local definitions. No old Rustlings or starter workflow appears in the rewritten lessons.

## Three purposeful browser tools

| Home | Tool | Deterministic arithmetic and control evidence |
|---|---|---|
| 01 | `neuron-fit` | Zero rule MSE9; weights/bias[1.5,.5] MSE.75 and final residual−1.5; boundary[−3,−3] MSE66; reset returns both controls/table/readout to zero model |
| 02 | `loss-gradient` | At zero analytical/numerical gradient[−8,−2], h1e-5 discrepancy<1e-6; atw1.3 h1e-16 discrepancy>.1; one simultaneous update[.8,.2] gives MSE3.52; dashed line honestly labeled slope guide, not the true probe secant |
| 04 | `classifier-evidence` | Sigmoid0=.5/BCEln2; slider±8 gives probabilities.999665/.000335 and wrong-target BCE8.000335; threshold.5 counts1/1/1/1; threshold.4 counts2/1/1/0; threshold1 reports undefined precision rather than equating it with zero; threshold0 counts2/2/0/0; reset restores every control |

Each tool is scoped to its container, contains labels/native controls, SVG title/description, numerical text/table equivalents, a deterministic reset, prediction prompts and no network or autoplay. Static lesson arithmetic remains usable without JavaScript. Tools explicitly illustrate rather than execute Rust; they never mark progress or mastery. Later chapters link to the existing tools rather than duplicating scripts.

`node labs/s01-foundations/demo_checks.js` executes the actual three scripts via built-in vm and a tiny DOM double, checking calculations and control/reset behavior. It is not a screenshot or screen-reader test; root's browser review covers presentation. The interactive researcher independently reviewed the demos and requested the precision-label and slope-guide corrections, which were applied and rechecked.

## Executed validation

All commands below completed successfully on2026-09-09:

- `cargo fmt --manifest-path labs/s01-foundations/Cargo.toml --check`.
- `cargo clippy --manifest-path labs/s01-foundations/Cargo.toml --all-targets -- -D warnings`.
- `cargo test --manifest-path labs/s01-foundations/Cargo.toml`: 11 tests passed, covering actual learner baselines, CLI validation, solution numerics and boundary cases. Chapter 06 checks the learner trainer's second L2 update against finite differences at a nonzero weight, and requires stopping after two worsening validation epochs while returning the epoch-zero model. Regression variants with a halved penalty or ignored patience are both rejected. Negative L2 is rejected even when the requested epoch count is zero.
- Each of01–06 baseline commands:exit0.
- Each of01–06 learner `--check`:exit1 with `GOAL_NOT_MET:` and numerical/structural learning-goal evidence.
- Each of01–06 `--solution --check`:exit0.
- All18 documented completed experiment variants:exit0.
- Eight invalid CLI/numerical-boundary cases:ordinary exit1 errors without a goal marker, including NaN/negative rates, invalid scale/threshold/candidate/batch, excessive steps and experiment options mixed with fixed goal checks.
- `node labs/s01-foundations/demo_checks.js`:all three scripts' calculations/bounds/resets passed.
- Local content audit:step order matches metadata; all exact original topic keys mapped; each session30–45min; unique IDs; all51 original section anchors preserved; no retired practice paths in lessons.

Measured default solution observations:01 fits both clean lines to displayed zero error;02 asymmetric initial gradient[−16,−7.333333...] with3 gradient row visits;03 unfamiliar scaled prediction error rounds to zero;04 BCE.009286;05 selected threshold.5 and testTP4,FP0,TN18,FN2, cost10 versus majority cost30;06 minibatch validation MSE.041615 and L2.008593. Full-batch SGD uses80 updates while the minibatch comparisons use240; these are not equivalent-work benchmarks. Tiny final-digit platform differences are tolerated.

## Research scope and limitations

The owner read the foundations/source inventory, learning-science synthesis and interactive research reports. Existing verified primary sources support mathematical conventions, feature scaling, evaluation leakage and optimization. Worked examples, retrieval, constructive explanation and controlled variations are research-informed design choices; no cited study establishes the efficacy of this complete self-directed Rust ML course.

The datasets and costs are disclosed teaching fixtures.05 intentionally preserves residual mistakes and exposes a forbidden post-outcome feature; correlated visits are not independent observations. Numerical success is implementation evidence, not real-world generalization or calibrated risk. The local checks are transparent rather than hidden; a source/self-explanation review remains necessary to assess whether numerical or analytical procedures were actually implemented honestly.
