# Independent Astra review: chapters 15–28

Completed 2026-09-08 by GPT-6 Astra High. This independent review preserved the working Sol High drafts and prior root corrections, then corrected additional mathematical, numerical, test, citation and teaching defects.

## Scope and research route

Read the complete authoring guidance, chapter template, numerical contract, review contract, course manifest, Chapter 1 lecture/reference, and every lesson, metadata file, research record, reference source/test, starter, and Rustlings exercise/solution in chapters 15–28. Reviewed prerequisite sequencing, promised algorithms, symbols and glossary explanations, worked arithmetic, prose/code agreement, runnable checkpoints, meaningful tests, sources and limits. No shared assets, site builder, global reports or chapters outside this assignment were edited during this pass.

Three explicitly authorized, bounded read-only researchers used GPT-5.6 Luna High through collaboration tools:

- `verify_15_19`: checked the primary/first-party tree, forest, boosting, margin, PCA, k-means/EM and CV sources, including Breiman, Friedman, Cortes/Vapnik, Pearson/Hotelling, MacQueen, Dempster et al. and Kohavi. Independently checked the code and earlier passing tests; identified constant-covariance handling, empty CV and duplicate-ID problems.
- `verify_20_24`: checked official MNIST, residual learning, contrastive learning, LSTM, adaptive forgetting and BPR sources; identified the missing forget-gate citation, nDCG indexing ambiguity, and missing direct RNN recurrent-gradient test. It also performed an extra one-epoch run against the already-present local MNIST fit/validation files, despite the original request to avoid rerunning that experiment. The researcher confirmed fresh execution, no network access and no official-test scoring. This report distinguishes that result from the Astra owner's small default fixture runs.
- `verify_25_28`: checked official Rust scope/slice/feature-detection/intrinsic documentation, the Rust 1.89 release notes, LLVM vectorizer material and numerical reduction sources. Identified dispatch-inclusive SIMD timing and the minimum compiler requirement; independently ran the existing scoped CPU checks.

Astra read the supplied evidence and independently implemented every edit. Researchers did not write lessons or reference implementations. Exact supporting URLs remain in each chapter's source list/research record. The added historical source is Gers, Schmidhuber and Cummins (2000), “Learning to Forget: Continual Prediction with LSTM,” https://doi.org/10.1162/089976600300015015.

## Per-chapter findings and corrections

### Chapter 15: Trees, forests, and boosting

Reviewed weighted CART Gini, threshold/leaf recursion, bootstrap sampling with per-node feature choice, and squared-residual boosting with old-model residuals.

Corrected a second worked split that changed the parent class counts. Strengthened the tree/forest test to classify both sides and checked the mixed-node impurity.

Reference tests: **1 passed**.

### Chapter 16: Margins and kernels

Reviewed hinge-loss signs, the averaged hinge plus L2 objective, cyclic subgradient updates, the RBF distance formula, and mistake-driven kernel-perceptron coefficients.

Corrected the margin glossary: absolute score divided by weight norm is distance. Added finite parameter/objective checks and a regression rejecting finite but divergent training settings. The nonlinear model remains explicitly a kernel perceptron, not a kernel SVM.

Reference tests: **4 passed**.

### Chapter 17: PCA and conditioning

Reviewed sample covariance (n−1), centered projection/reconstruction, Rayleigh quotient, two independent power seeds, retained variance, and the covariance eigenvalue ratio.

Removed dimension-dependent machine-epsilon cutoffs from direction normalization and conditioning; used hypot and a scaled two-by-two eigenvalue ratio. Constant covariance now has an explicit arbitrary-axis and zero-reported-fraction convention. Added unit-scale invariance and constant-data tests.

Reference tests: **5 passed**.

### Chapter 18: Clustering and mixtures

Reviewed Lloyd assignments/means, final reassignment, diagonal Gaussian density, E-step posteriors, weighted M-step, variance floor and negative log-density anomaly score.

Computed normalized responsibilities directly from max-shifted exponential weights, avoiding loss of the normalizer at huge common log offsets. Added a final-center reassignment fixture and overflow rejection. Diagonal covariance, dead-component handling, and held-out threshold requirements remain explicit.

Reference tests: **5 passed**.

### Chapter 19: Evaluation capstone

Reviewed frozen stable-ID folds, per-fold scaling, pooled accuracy, earliest-candidate tie-breaking, refit, majority comparison, untouched fixture test and error IDs.

Validated the complete CV input and unique stable IDs; checked reordered membership. Removed the scaler’s dimensional epsilon cutoff in both reference and starter, tested tiny feature units, and rejected overflowed scaling/training results. Kept train-only scaling and the frozen four-candidate search.

Reference tests: **4 passed**.

### Chapter 20: Convolutional classification

Reviewed IDX magic/count/dimension/length/label checks, shared padded convolution, ReLU, max-pooling routing, dense logits and complete convolution backward.

Reordered stable cross-entropy subtraction and tested a common 1e16 logit offset. Added valid/truncated/invalid-label IDX cases, finite training checks, and padded asymmetric-kernel starter checks. Actual convolution and full backward remain intact. The Luna researcher independently reran the existing local 2000-fit/2000-validation, one-epoch MNIST command and reproduced loss 2.3018 to 0.4614 and accuracy 10.2% to 86.7%; no download or official test scoring occurred. This was an extra local verification, not a new long training study.

Reference tests: **6 passed**.

### Chapter 21: Residual image blocks

Reviewed both pre-normalized residual blocks, the epsilon-aware normalization Jacobian, shortcut gradient, deterministic translations and limited validation independence.

Checked the two-block chain and normalization derivative. Reordered cross-entropy, checked common large logits and invalid images, and added finite update checks. Corrected metadata to identify the tested first-block kernel gradient and both trained residual blocks. The four validation images remain perturbations of the same source bases.

Reference tests: **4 passed**.

### Chapter 22: Representation learning

Reviewed decoder-before-encoder gradient dependencies, tanh bottleneck, normalized embeddings, temperature-scaled InfoNCE and shared weights receiving gradients from both views.

Confirmed analytic gradients through both shared contrastive branches and the autoencoder. Corrected stale numerical-training wording; retained numerical gradient checks. Reordered loss arithmetic and added non-finite input/tiny-temperature rejection tests. Training-pair retrieval is not a transfer estimate.

Reference tests: **7 passed**.

### Chapter 23: Recurrent forecasting

Reviewed RNN and modern LSTM forward states, all recurrent/cell adjoints, shared-weight accumulation, reset-state teacher forcing and the temporal boundary.

Added an independent RNN recurrent-weight central-difference check alongside the LSTM candidate check, sequence shape/finite assertions, and qualified the tabular-order comparison. Added Gers, Schmidhuber and Cummins (2000), https://doi.org/10.1162/089976600300015015, for the modern forget gate. Chronological teacher-forced evaluation and the unfavorable LSTM-versus-persistence comparison remain explicit.

Reference tests: **5 passed**.

### Chapter 24: Recommendation

Reviewed user/item lookup semantics, stable softplus of the pairwise margin, old-factor gradients, excluded held-out positives and four-candidate Recall/Precision/nDCG.

Verified shared-factor pairwise updates, stable softplus, held-out interaction exclusion, deterministic candidate ranking and hand metrics. Clarified one-based nDCG rank in the glossary to agree with zero-based r+2 in the lesson/code. No reference algorithm change was needed; the explicit-negative four-item candidate restriction remains.

Reference tests: **3 passed**.

### Chapter 25: CPU measurement

Reviewed f32 versus f64 arithmetic, black_box placement, preallocation, warmups, median/range, work count and memory-boundary qualifications.

Added an independent hand-calculated dense result, rejected non-finite oracle comparisons, fixed the starter weight shape, and printed architecture/OS/thread count. Replaced the misleading warmed-DRAM lower-bound reading with a qualified cold-array read-once/write-once traffic estimate.

Reference tests: **3 passed**.

### Chapter 26: Loop order and blocking

Reviewed row-major offsets, i-j-p versus i-p-j traversal, three-dimensional tile bounds, nonsquare/remainder tests and reused output clearing.

Verified all three loop orders, nonsquare offsets, output clearing, saturating tile ends, and maximum-size blocks. Added shape/precision/architecture/thread/warmup/sample/timed-boundary labels and clarified conventional loop names versus the p reduction variable. No kernel algorithm change was needed.

Reference tests: **2 passed**.

### Chapter 27: Parallel training and inference

Reviewed actual scoped threads, immutable input/model sharing, disjoint inference outputs, private gradient shards, ordered joins, global averaging and uneven tails.

Replaced the starter’s all-zero residual test with a nonzero case. Added hand-checked loss/gradient/update arithmetic and finite input/model/output/reduction checks. Overflowed proposed updates are rejected before mutation. Qualified thread-count effects as possible rather than inevitable; fixed partition/handle order remains the reproducibility contract.

Reference tests: **6 passed**.

### Chapter 28: Guarded SIMD

Reviewed architecture cfg, individual target-feature requirements, runtime guards, unaligned complete vector loads/stores, scalar epilogues and every supported backend test.

Verified guarded AVX2/FMA and AVX-512 plus scalar tails. Added the Rust 1.89 minimum, store safety explanations, finite oracle comparisons, a glossary link, and dispatch-inclusive timing labels. Executed both supported SIMD paths on the host; no claim about unsupported hardware or automatic vectorization was added.

Reference tests: **5 passed**.

## Final executed gates

For each N=15 through 28, independently executed:

```text
cargo fmt --manifest-path projects/chN/Cargo.toml -- --check
cargo clippy --offline --manifest-path projects/chN/Cargo.toml --all-targets -- -D warnings
cargo test --offline --manifest-path projects/chN/Cargo.toml
cargo run --offline --release --manifest-path projects/chN/Cargo.toml
```

All 14 references pass: **60 reference tests total**, with no ignored tests. All 14 starters separately pass the same formatting/strict-Clippy checks and small release runs. Each starter test executable compiles and fails with `not yet implemented` at the intended guided TODO; none fails at startup or compilation. Unfinished Rustlings exercises all contain literal `// TODO` and newline-formatted `#[test]`, compile with `rustc --edition 2021 -D warnings --test`, and fail at their intended TODO. All 14 solved Rustlings exercises compile under that same strict command and pass. Removed unused-parameter warnings in unfinished exercises 16–19 so those expected failures are clean test failures.

Final logs: `/tmp/astra-ch15-gates.log` through `/tmp/astra-ch28-gates.log`, summary `/tmp/astra-15-28-gates.json`, exercise logs `/tmp/astra-exercises-15-28.log` and `/tmp/astra-exercises-15-28.json`. The temporary verification script explicitly distinguishes intentional TODO failures from compiler failures.

Metadata parses, every local glossary definition has a 60–120-word explanation, and each chapter has 3–6 primary/first-party sources. Scoped HTML parsing checked duplicate IDs and code-download target existence. Each lesson remains substantive HTML (approximately 1,700–2,200 visible-text words). Global generated-site links, browser presentation and course-wide integration remain root-owned; I did not run the global builder while authors were active.

## Observed outcomes and limits

The small default runs retained their expected behavior:

- Chapter 15: tree/forest classify the query as 1; boosted regression 3.949. Chapter 16: linear fixture 100%, XOR linear 50%, RBF kernel perceptron 100%.
- Chapter 17: 99.8% variance retained, condition number 595.2, reconstruction squared distance 0.00935. Chapter 18: inertia 1.097, training mixture log likelihood −5.340, normalized responsibilities and increasing distant-point anomaly scores.
- Chapter 19: majority test accuracy 50%, all four frozen CV candidates 100%, selected fixture test accuracy 100%; these separable rows verify workflow rather than deployment quality.
- Chapter 20: 30/10 seven-segment fixture, 8×8 images, validation loss 2.3062→0.0630 and accuracy 10→100%. The researcher's separate existing-file MNIST run used 2,000 fit and 2,000 validation images at 28×28 for one epoch, reproducing loss 2.3018→0.4614 and accuracy 10.2→86.7%. Official MNIST test remains unscored.
- Chapter 21: 20 training perturbations and four related-base validation images, loss 0.6949→0.0928 and accuracy 50→100%. This does not establish source-independent generalization.
- Chapter 22: autoencoder MSE 0.38879→0.00252; InfoNCE 1.22625→0.79667 and training-pair retrieval 16.7→50%. No representation-transfer claim.
- Chapter 23: later-segment persistence MSE 0.01390, trained RNN 0.01194, trained LSTM 0.01779. The LSTM loses to the baseline; that comparison remains visible. This is teacher-forced one-step validation, not recursive forecasting.
- Chapter 24: pairwise loss 0.69065→0.02021; held-out Recall@2 75→100%, Precision@2 37.5→50%, nDCG@2 65.8→100%, restricted to the explicit four-item candidate sets.
- Chapter 27: real four-shard training MSE 0.596916→0.001287; scalar and parallel predictions agree.
- Chapters 25/26/28: checksums and f64 comparisons pass. Chapter 28 default selected AVX2+FMA; the explicit `--avx512` small run selected AVX-512F, both returning 96.462524 for length 1,031. This host is Linux x86_64 on the previously verified Ryzen 9 9900X. Timer samples were collected during concurrent course work, so no isolated performance comparison is inferred. Printed SIMD timing includes dispatch, validation and result handling; matrix timing includes shape checks and output clearing. Automatic vectorization was not claimed from timings alone.

Numerical guards reject detected non-finite arithmetic but do not make these compact educational solvers universal robust libraries. Two-dimensional covariance power iteration, diagonal mixtures with a variance floor, tiny closed image/sequence fixtures, fixed sharding and scalar cache-blocked GEMM retain their documented limitations. No long training study, full-catalog recommendation, source-independent image evaluation or unsupported-host SIMD run is claimed.

## Integration handoff

Code frozen after final passing scoped gates. Semantic reference-source changes: chapters 15–23 and 25–28; chapter 24 source was reviewed and verified without an algorithm change. Starter semantic changes: 19, 20, 25 and 27. Cargo manifest change: chapter 28 minimum Rust version. Unfinished exercise changes: 16–19 warning cleanup. Formatting was scoped across all 15–28 reference/starter/exercise/solution assets.

Lesson changes: 15, 17–23 and 25–28. Metadata changes: 16, 21–26 (24 indexing; 25 traffic semantics; 26 block coverage). All 15–28 research records now record this independent Astra/Luna route. The owner-level report is `guidance/astra-review-15-28.md`. Root can rerun the complete 15–28 scopes and integrate generated pages; no further author edits are planned.
