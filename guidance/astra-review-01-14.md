# Astra High review: Chapters 01–14

Reviewed 2026-09-08 by GPT-6 Astra High. Bounded GPT-5.6 Luna High researchers independently checked primary-source claims and worked arithmetic for 01–07 and 08–14; they did not edit chapters. The Astra reviewer read every owned lecture, metadata/research file, reference implementation and tests, starter, Rustlings exercise/solution, and Chapter 01's browser-demo source. The authoring contract, chapter template, numerical conventions, review requirements, course topic map, and full Chapter 01 reference informed the review.

This is a substantive review, not a gate-only approval. The reference behavior below was executed locally. Website build/navigation/render checks remain the lead's integrated responsibility.

## Corrections and evidence by chapter

### 01 — One neuron learns from data

The five-row MSE is 9 at the origin; numerical gradients produce weight 0.8 and bias 0.2, reducing loss to 3.52. The reference's simultaneous update and browser arithmetic agree. After 100 steps, the printed parameters are 2 and 1; predictions at −1.5, 0.5, and 3 are −2, 2, and 7. The interpolation/extrapolation and synthetic-data limitations are accurately stated.

Strengthened the starter's prediction test with a second model/input, so a hard-coded fixture prediction fails. No reference or interactive calculation was changed. [Google's own regression documentation](https://developers.google.com/machine-learning/crash-course/linear-regression) supports the weight/bias/loss framing. The Rust Book and Rustlings documentation are appropriate implementation/workflow sources, not evidence for real-world predictive quality.

### 02 — Why learning works

Verified the chain rule, factor two for ordinary MSE, averaging, simultaneous update, and combined absolute/relative tolerance. Gradients at zero are (−8,−2). The asymmetric variation gives (−16,−22/3). Both reference finite-difference settings pass.

Strengthened starter and Rustlings checks at nonzero weights and with unequal row contributions, preventing a constant gradient or one-row derivative from passing. [PyTorch gradcheck](https://docs.pytorch.org/docs/main/generated/torch.autograd.gradcheck.gradcheck.html) supports independent numerical comparison and its limits; MIT calculus supplies the derivative foundation. No reference formula correction was needed.

### 03 — Multiple inputs

Verified the worked dot product 127, reordered-feature result 301, one-row gradient [−28,−42] with bias −14, and standardized counts [1,2,3] with deviation √(2/3). The measured reference MSE is 2.0695 and held-out prediction is 242.57.

The interaction exercise ambiguously listed coefficients [2,3,4] immediately before requiring scaling. It now states that these are raw-feature coefficients and explains the transformed weights and intercept. Added central differences for all three implemented weight gradients and bias. [StandardScaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) supports training-only fitted statistics. Coefficients remain predictive associations, not causal effects.

### 04 — Classification

Verified logit 1.5 → probability 0.8176, BCE values, the stable sign-split sigmoid, the p−y derivative, and the update to logit 0.30. The reference's BCE falls from 0.693147 to 0.009286 and all fixture rows classify correctly.

The shared `logit` glossary definition was binary-only, yet later chapters reuse the slug for multiclass scores. It now distinguishes binary log-odds from relative softmax scores, where positivity alone does not imply probability above one half. Learner checks now require actual sigmoid values and BCE for neutral and confidently wrong logits; returning any finite constant no longer passes. [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.BCEWithLogitsLoss.html) supports the stable combined objective.

### 05 — Measuring learning

Verified test counts TP=1, FP=1, TN=7, FN=1; accuracy 0.8, precision/recall/F1 0.5, balanced accuracy 0.6875. Validation selects 0.6 with documented tie handling. Entity overlap detection and validation-only threshold selection remain intact.

The prose promised visible counts but the executable printed only metrics. The program now prints both model and baseline confusion counts before their metrics. [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) and [Google threshold guidance](https://developers.google.com/machine-learning/crash-course/classification/thresholding) support the evaluation boundary. Supplied scores and tiny samples remain clearly distinguished from trained-model quality evidence.

### 06 — Reliable optimization

Verified bowl arithmetic (loss 4 → 2.56 at rate 0.1, but 4 → 5.76 at 1.1), batch contributions −2 and −6, and L2 total gradient −2.6. Measured steady validation MSE is approximately 0.025; the shown L2 setting gives approximately 0.007, without claiming universal improvement.

Invalid rate or L2 values were accepted when epochs=0 because validation only ran inside the update loop. Training now validates those arguments before the loop. Added regression checks for zero epochs, one-example batch averaging, and leaving bias unpenalized. Learner tests now cover zero regularization and negative weights. [Bottou's SGD treatment](https://leon.bottou.org/papers/bottou-2010) and [Hoerl–Kennard ridge regression](https://doi.org/10.1080/00401706.1970.10488634) support the method distinctions. Fixed ordered minibatches remain an explicit teaching limit.

### 07 — From a neuron to XOR

Verified the 2–2–1 shapes, all nine parameters, forward hidden values approximately 0.6900/0.3775, output 0.6285, and backward contributions. Training yields BCE 0.001962 and probabilities approximately [0.0017,0.9977,0.9977,0.0015].

The displayed per-example derivatives used the symbol for mean loss; these now use ℓ and explicitly distinguish the averaged L. Defined tanh before the transfer task. Existing checks independently recalculated two analytical formulas; an additional test now compares **the actual implemented update** against central differences for every parameter, exposing update-order, sign, reduction, and hidden-path defects. Sigmoid-slope learner tests include values away from 0.5. The [Stanford PDP XOR treatment](https://web.stanford.edu/group/pdplab/pdphandbookV3/handbookch6.html) and [1986 backpropagation paper](https://doi.org/10.1038/323533a0) support the mathematical account.

### 08 — Automatic differentiation

Verified graph identity, consumer-before-parent traversal, shared operand accumulation, fresh backward semantics, and the tanh chain rule. x²+x at x=3 returns 12 with gradient 7; repeating backward on x² at x=2 leaves gradient 4.

A finite final value could hide nonfinite graph data: tanh(MAX×2) returns 1. Backward now rejects nonfinite reachable intermediates and overflowing adjoints. Tests also cover a finite zero output whose derivative overflows. Learner accumulation tests inspect the first contribution before adding the second. [Baydin et al.](https://jmlr.org/papers/volume18/17-468/17-468.pdf) and [micrograd](https://github.com/karpathy/micrograd) support reverse accumulation; the original Rust code remains independent. Scalar heap nodes, single-threaded borrowing, and recursive traversal are acknowledged limits.

### 09 — Tensors and batches

Verified every hand-computed result: Y=[−1.5,3.5,−1.5,12.5], dX=[5,2,−1,11,4,−3], dW=[13,17,21,18,24,30], db=[4,6]. The nonsquare shapes expose transposition errors, and the weight finite-difference check passes. Upstream derivatives own any averaging factor; the layer adds no hidden extra average.

No reference correction was needed. Strengthened learner offset tests with several row/column/width combinations. [NumPy matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html), MIT matrix calculus, and [JAX VJP documentation](https://docs.jax.dev/en/latest/jacobian-vector-products.html) support the layout and derivative account. The kernel remains a small scalar reference, not an optimized tensor library.

### 10 — Recognizing digits with a linear model

Verified IDX headers, checked payload arithmetic, class bounds, pair alignment, normalized pixels, ten-class shapes, gradient averaging, and the authentic generated-data training path. The fixture's held-out loss falls 2.3026 → 0.0133 and accuracy 0% → 100%.

The old expression `log_sum_exp(z) - z[target]` lost precision after adding a large common maximum. Cross-entropy now cancels that offset before adding the logarithmic term. Ten equal logits at 1e16 correctly give ln 10 instead of 2. Added this regression and parser checks for truncation, label range, and count mismatch. Strengthened softmax/log-sum-exp learner checks. The lesson explains the numerical ordering. [Official MNIST](https://yann.lecun.org/exdb/mnist/) supports format/count facts; [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) supports the log-softmax/NLL relationship. The previously measured 55k/5k MNIST results were preserved, not rerun.

### 11 — An MNIST neural network

Verified ReLU forward/backward, width-aware initialization, momentum's two steps to 0.8 and 0.62, Adam moments/bias correction, and strict checkpoint structure. The default fixture reaches held-out loss about 0.0029 and 100% accuracy, then restores optimizer step 300. Existing next-step exact-resume checks pass.

The lesson instructed epoch 0 for frozen final evaluation, but code rejected it. Zero epochs now require an existing checkpoint, evaluate it without updates, and do not rewrite its bytes. A regression checks byte preservation and rejects missing checkpoints. Train and held-out image dimensions must match individually, not merely have equal pixel counts. Cross-entropy cancellation is fixed in both reference and useful SGD starter. Nonfinite loss/gradients and overflowed Adam moments now produce errors; finite parameters alone do not prove valid optimizer state. The momentum starter now checks a second step, exposing implementations that discard memory. [Adam](https://arxiv.org/abs/1412.6980) and [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) support the required state. Existing checkpoint optimizer choices still intentionally override creation-time CLI defaults.

### 12 — Statistics and uncertainty

Verified observed accuracy 0.75, replacement values 0.625/0.875, bootstrap examples 0.50 and 1.00, calibration mean 0.7625 versus frequency 0.75, and Brier examples 0.04/0.34. The executable returns interval [0.375,1.000] and Brier 0.1794 for its eight observations.

The bootstrap test previously checked only repeatability, which a constant interval could pass. It now checks all-correct, all-wrong, and mixed samples, exact calibration summaries, and malformed predictions. Learner Brier tests include unequal errors. The already-correct repeated-sampling interval interpretation was preserved. [Efron's paper](https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full), [calibration guidance](https://scikit-learn.org/stable/modules/calibration.html), and [Austin–Steyerberg](https://pubmed.ncbi.nlm.nih.gov/24895046/) support the claims. Eight-row percentile coverage is not guaranteed; grouped/block bootstrap is explained, not implemented.

### 13 — Real tabular data

Verified training-only median/vocabulary, missing indicators, sorted deterministic columns, unknown-category behavior, group isolation, and temporal cutoff semantics. The default median is 12 and the unknown held-out row is [100,0,0,0]. The deliberate held-out 999/secret leakage fixture still passes.

The even-count median `(a+b)/2` overflowed for two finite MAX values. It now uses native `f64::midpoint`, stable since Rust 1.85, with a finite-extreme regression. Missing indicators, all-missing fitting, and nonfinite transforms are also tested. The learner median test now uses a second unsorted odd-length fixture. [Rust midpoint documentation](https://doc.rust-lang.org/std/primitive.f64.html#method.midpoint) and official [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html), and [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) support the implementation choices. The project deliberately handles one numeric and one categorical feature.

### 14 — Nearest neighbors and naive Bayes

Verified squared distance 25, standardization examples, class-vote tie policy, Gaussian contribution approximately −1.419, priors/means/variances, and both models choosing the middle cluster.

Gaussian prediction accepted NaN queries and returned an arbitrary winning class. It now returns a checked result and rejects nonfinite coordinates and scores. Scaling/Gaussian fitting reject overflowed statistics; kNN rejects overflowed distances before sorting. Regression checks cover these cases, a hand-computed two-feature Gaussian score, and a tied vote. Distance learner tests now include zero distance and a single changing coordinate. The glossary clarifies that a constant training coordinate provides no neighbor discrimination, rather than claiming every future query contributes zero distance. [Cover–Hart](https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf), [nearest-neighbor guidance](https://scikit-learn.org/stable/modules/neighbors.html), and [Gaussian NB guidance](https://scikit-learn.org/stable/modules/naive_bayes.html) support the method account. The six-row fixture remains a mechanism check; probabilities are not normalized or calibrated by this implementation.

## Executed gates

For every chapter NN in 01–14:

- Reference: `cargo fmt --manifest-path projects/chNN/Cargo.toml --check`; `cargo clippy --manifest-path projects/chNN/Cargo.toml --all-targets -- -D warnings`; `cargo test --manifest-path projects/chNN/Cargo.toml`; `cargo run --quiet --manifest-path projects/chNN/Cargo.toml` — all pass. There are 44 passing reference tests in total.
- Starter: the same fmt/clippy gates and useful default run pass. `cargo test` exits 101 at the intentional `todo!`, as required.
- Rustlings exercise and solution: `rustfmt --edition 2021 --check`; `rustc --edition 2021 --test` compilation; execute the generated test binary. All solutions pass and all unfinished exercises fail at `todo!`.
- Temporary solved copies of all 14 starter functions compile and pass their strengthened checks. These copies live only under `/tmp`; repository starters remain unfinished.

Final focused follow-up gates were run after the last changes to 06, 08, and 14. Every exercise retains literal `// TODO` and the required newline after `#[test]`. Metadata parses and owned HTML structural checks pass. Raw local gate output is stored in `/tmp/astra-review-01-14-gates.json`, `/tmp/astra-review-followup-gates.json`, and `/tmp/astra-review-solved-starters.json` for this review session.

## Scope and remaining limits

No global builder/verifier, shared asset mutation, download, official-test scoring, or long training was performed. The lead's prior MNIST 55k/5k smoke measurements remain explicitly attributed to those earlier runs. Final rendered-site navigation, responsive layout, and integrated glossary behavior need the lead's integrated checks. None of the synthetic fixtures establishes real-world quality, exact universal numerical behavior, or production robustness.
