# Section 03 — statistics and classical tabular learning

Status: implemented and author-validated on 2026-09-09. Chapters 12–19 use one independent standard-library Rust package, `labs/s03-tabular`. All eight original reference projects remain unchanged. The three assigned browser tools are housed once in Chapters 12, 13 and 14, with later chapters linking to the shared geometry tool.

## Eight-chapter design matrix

Each session totals 40 minutes; implementation and transfer are described in the session-specific metadata and prose. Native hint/answer disclosures provide progressive help. Goals below refer to actual learner functions, not printout matching.

| Chapter / original topics | Observable objectives and explanation steps | Meaningful learner change | Goal evidence and transfer | Sessions |
| --- | --- | --- | --- | --- |
| 12: Sampling; confidence intervals; bootstrap; calibration | Distinguish population claims, sample estimates and independent sampling units; Implement and interpret a whole-machine percentile bootstrap interval; Compute counted calibration bins and distinguish probability error from decision accuracy<br>Sampling → `sample-estimate`; confidence intervals → `bootstrap`; bootstrap → `bootstrap-build`; calibration → `calibration` | Replace row-normal approximation by group-index construction, repeated whole-machine draws and empirical percentiles; replace a global bin with counted reliability bins. | Degenerate/all-correct and duplicated-within-machine cases; unfamiliar bins; change the resampled statistic to Brier or sharpen fixed odds. | 3 ×40 min |
| 13: Missing values; categories; grouped splits; temporal splits | Fit missing-value imputation and standardization using training observations only; Encode known and unknown categories with a fixed training vocabulary; Implement grouped and temporal splits that match the deployment question<br>Missing values → `missing`; categories → `categories`; grouped splits → `structured-splits`; temporal splits → `availability` | Fit median, imputed mean/scale and sorted vocabulary; construct all four raw-row membership policies. | Fit checkpoint first, then exact ID membership for all four policies; held-only extreme/category perturbation; same-machine temporal claim. | 3 ×40 min |
| 14: Distance; scaling; nearest neighbors; Gaussian naive Bayes | Calculate full-vector distances and explain the effect of feature scaling; Implement deterministic nearest-neighbor voting for general k; Estimate Gaussian class priors, means and variances and explain conditional-independence limits<br>Distance → `distance`; scaling → `scaling`; nearest neighbors → `knn`; Gaussian naive Bayes → `gnb` | Store/rank full-vector distances and vote over k neighbors; fit class frequencies and per-class feature variances. | Nearest label differs from k3 majority; unequal priors/variances and a constant feature; matched vibration-column ablation. | 3 ×40 min |
| 15: Decision trees; bagging; random forests; gradient boosting | Implement recursive impurity-based decision trees with explicit stopping rules; Train genuinely resampled bagging and per-node random-feature forests; Fit squared-residual gradient boosting and compare training with validation complexity<br>Decision trees → `tree`; bagging → `forest`; random forests → `forest-build`; gradient boosting → `boosting` | Grow recursive CART, generate actual bootstrap trees and per-node feature subsets, fit sequential squared-residual stumps. | Recursive XOR behavior, non-cloned ensembles and residual fitting; vary tree depth, ensemble seeds/features and shrinkage/rounds in named report values. | 4 ×40 min |
| 16: Hinge loss; linear SVM; kernel similarity | Calculate signed margins and implement active/inactive hinge updates; Train a regularized linear SVM and explain its weight penalty; Calculate RBF kernel similarity and fit a kernel perceptron while distinguishing it from a kernel SVM<br>Hinge loss → `hinge`; linear SVM → `linear-training`; kernel similarity → `kernels` | Implement active/inactive regularized hinge updates and learn all kernel-perceptron coefficients. | Known active/inactive update and linearly separable fixture; inward-moved XOR; changed gamma and regularization. | 3 ×40 min |
| 17: Covariance; eigenvectors; power iteration; projections; conditioning | Compute centered full covariance with declared shapes and denominator; Recover orthogonal eigenvectors with bounded power iteration; Project and reconstruct rows with a training-fitted PCA model; Diagnose conditioning, eigengap sensitivity and representation-dependent variance<br>Covariance → `covariance`; eigenvectors → `eigenvectors`; power iteration → `power`; projections → `projections`; conditioning → `conditioning` | Replace diagonal-only coordinate compression by full covariance and orthogonal power components; implement eigenpair residual diagnostics. | Known covariance eigenvalue/residual and orthogonality; translated rows, rank-one3D and constant data;20 versus400 iterations on explicit near-tied rows. | 4 ×40 min |
| 18: K-means; Gaussian mixtures; EM; anomaly scores | Implement k-means assignment and mean-update iterations; Fit weighted diagonal Gaussian mixture components through stable EM; Calculate responsibilities and negative-log-density anomaly scores; Explain initialization, covariance and projection limits on clustering claims<br>K-means → `kmeans`; Gaussian mixtures → `mixtures`; EM → `maximization`; anomaly scores → `anomaly` | Alternate assignments and means, then frozen E-step and weighted M-step with variance floors; add entropy/shift diagnostics. | Known means/inertia and final reassignment; unequal spreads, normalized huge-offset responsibilities and monotone likelihood; shifted distant query and changed variance floor. | 4 ×40 min |
| 19: Cross-validation; hyperparameter search; fair baselines; error analysis | Construct reproducible group/time cross-validation with fold-local preprocessing; Search a declared hyperparameter budget using pooled cost and stable ties; Compare three learned model families with training-only majority/prevalence baselines; Inspect final residuals and slices without reselecting on final outcomes<br>Cross-validation → `cross-validation`; hyperparameter search → `search`; fair baselines → `fair-baselines`; error analysis → `error-analysis` | Rotate machine blocks under chronology; evaluate all six candidates, pool errors and select by fixed cost; add a descriptive final-day slice. | Three48/18 folds,54 distinct validation IDs and six candidates; reverse source rows; cost reconstruction; new-cycle protocol for existing machines. | 3 ×40 min |

## Actual session route

### Chapter 12

- Session 1 (40 minutes): Build a paired-row bootstrap and explain seed variation on an unfamiliar mixed sample. Steps: `sample-estimate`, `bootstrap`.
- Session 2 (40 minutes): Resample whole machines and explain why within-machine duplication adds no independent evidence. Steps: `cluster-sampling`, `bootstrap-build`.
- Session 3 (40 minutes): Build counted reliability bins and compare confidence sharpening with fixed decisions. Steps: `calibration`, `brier-rust`.

### Chapter 13

- Session 1 (40 minutes): Fit medians and numeric scales from training observations; handle a changed missing/zero fixture. Steps: `split-first`, `missing`.
- Session 2 (40 minutes): Fit a category vocabulary and encode a held-only category without changing width. Steps: `categories`, `rust-project`.
- Session 3 (40 minutes): Implement four split policies and adapt the development split to a same-machine future claim. Steps: `structured-splits`, `availability`.

### Chapter 14

- Session 1 (40 minutes): Implement deterministic k-neighbor voting and predict a changed majority after scaling. Steps: `distance`, `scaling`, `knn`.
- Session 2 (40 minutes): Fit unequal Gaussian class priors and spreads and check local unfamiliar queries. Steps: `gnb`, `bayes-build`.
- Session 3 (40 minutes): Implement a matched sensor ablation and compare assumptions and storage/query costs. Steps: `compare`, `classical-geometry`.

### Chapter 15

- Session 1 (40 minutes): Build recursive CART splits and test depth on a nonlinear fixture. Steps: `tree`, `tree-build`.
- Session 2 (40 minutes): Build resampled bagging and random-feature forests and compare genuinely different tree predictions. Steps: `forest`, `forest-build`.
- Session 3 (40 minutes): Fit residual stumps for squared-loss boosting and trace errors after changed rounds. Steps: `boosting`, `boost-build`.
- Session 4 (40 minutes): Implement matched model comparisons and explain why greater fitted complexity can worsen validation. Steps: `experiment`, `practice`.

### Chapter 16

- Session 1 (40 minutes): Implement hinge loss/update branches and distinguish a correct low-margin case from an inactive update. Steps: `boundary`, `hinge`.
- Session 2 (40 minutes): Fit the regularized linear SVM and compare weight norm and decisions at changed regularization. Steps: `linear-training`, `margin-audit`.
- Session 3 (40 minutes): Fit kernel-perceptron coefficients and test inward-moved XOR queries at varied gamma. Steps: `kernels`, `experiment`.

### Chapter 17

- Session 1 (40 minutes): Fit centered full covariance and verify translation invariance. Steps: `covariance`, `covariance-build`.
- Session 2 (40 minutes): Implement power iteration and check known eigenpairs plus rank-one three-dimensional inputs. Steps: `eigenvectors`, `power`.
- Session 3 (40 minutes): Retain orthogonal components and verify multicomponent reconstruction after translation. Steps: `projections`, `multi-component`.
- Session 4 (40 minutes): Implement residual diagnostics and distinguish slow iteration from near-tied-axis sensitivity. Steps: `conditioning`, `review`.

### Chapter 18

- Session 1 (40 minutes): Implement Lloyd assignments/means and test an extreme added reading and final reassignment. Steps: `kmeans`, `center-updates`.
- Session 2 (40 minutes): Build a frozen E-step responsibility matrix and transfer equal-mixture normalization to three components. Steps: `mixtures`, `expectation`.
- Session 3 (40 minutes): Implement weighted M-step parameters and compare responsibility entropy on changed queries. Steps: `maximization`, `mixture-check`.
- Session 4 (40 minutes): Implement a local shifted-query anomaly diagnostic and interpret floor-dependent density scores. Steps: `anomaly`, `anomaly-review`.

### Chapter 19

- Session 1 (40 minutes): Implement legal rotating machine/time folds and preserve membership under reversed source order. Steps: `protocol`, `cross-validation`.
- Session 2 (40 minutes): Evaluate all frozen candidates and verify pooled costs against raw error counts. Steps: `search`, `fair-baselines`.
- Session 3 (40 minutes): Open the frozen final report, add a descriptive day-11 slice and specify a new-cycle temporal protocol. Steps: `final-evaluation`, `error-analysis`.

## Data, continuity and evaluation

The controlled maintenance-v1 dataset contains 288 rows, 24 machines and 12 days, generated by the disclosed `data/generate.py` with seed 20260909. The stored CSV fingerprint is FNV-1a64 `6dccde7b74debd05`. Sensor correlations arise from shared wear; machine offsets create dependence; Bernoulli labels create overlap; later measurement/label drift and a novel turbo category produce a genuine shift. Some temperature missingness depends on the outcome process. The generator is provenance, not a source of fresh evaluation samples after inspection. Its final version was frozen before the comparison.

Chapter 12 evaluates a frozen sensor heuristic on 18 development readings. Chapter 13 changes the task into actual train-only preparation; Chapters 14–16 fit classifiers; Chapter 17 studies sensor projection; Chapter 18 fits operating-regime density; Chapter 19 returns to supervised failure prediction. The consistent public row contains stable identity, machine, prediction day, three optional numeric sensors, regime, label, late repair field and the heuristic. IDs/time/late repair/heuristic are excluded from learned features. Completed preparation yields nine features (3 numeric +3 missing flags +2 vocabulary bits +1 unknown bit). PCA intentionally takes only the three numeric columns; the clustering representation has two fitted PCA coordinates.

Regular training is 48 early rows from machines0–11 through day3, with day4 gap and18 validation rows from machines12–17 on days5–7. Development is144 rows from machines0–17 through day7. The capstone rotates three blocks of six machines, each time fitting48 early rows and predicting18 later held-machine rows. Thus54 eligible future validation rows receive one prediction each; early/gap rows are intentionally not validation candidates. Final refit uses all144 development rows, followed by day8 gap and18 final rows on machines18–23, days9–11. Label availability is day+1 throughout.

The capstone actually fits kNN, Gaussian NB and recursive CART, with two configurations per family. Frozen criterion is pooled `(4FN+FP)/n` at threshold.5, earliest candidate on ties. Ordinary runs and goal checks evaluate only development. Explicit `19 --final` opens final metrics, per-family finalists, training-majority/prevalence baselines, whole-machine uncertainty intervals and residual slices. Final outputs are a teaching evaluation boundary, not file access control. Inspection begins a later development cycle; it cannot support reselection on the same test. Root performs the final report run for integrated delivery evidence.

## Verified calculations and observations

Author validation runs actual learner and solution functions and records outputs in `guidance/validation-redesign/section-03-author.txt` with32 machine-readable CLI statuses in `section-03-author.json`. Key completed-development observations:

- Chapter12:12/18 correct, Brier.1859, whole-machine bootstrap seed7/B2000 endpoints[.5000,.8333]. Four bins have counts[5,6,5,2], average probabilities[.1431,.4044,.6508,.8255] and frequencies[.2,.3333,.4,1]. The differing row-normal interval is a different estimator; the lesson does not falsely promise the group interval is always wider.
- Chapter13:row split26/36, grouped33/48, time36/54, strict12/18. Forbidden late repair gives18/18 while violating availability. Strict train-only temperature mean61.137 differs from fit-before-split61.721. Exact membership is checked for all four policies after the numeric/vocabulary milestone.
- Chapter14:k1 gives10/18; k3 and k7 each12/18, with Brier.2346 and.2166. Gaussian NB at floor.05 gives12/18, Brier.2554. Same accuracy does not imply equal probability quality.
- Chapter15:depth1/3/6 trees have3/11/25 nodes and validation12/11/10 correct of18. Boosting1/10/30 rounds gives Brier.2342/.2047/.2178. More training complexity can worsen validation.
- Chapter16:linear SVM lambda.01/.2 yields13/18 and12/18; norms about1.647 and.514. RBF kernel-perceptron gamma.05/.5/5 yields7/11/10 correct of18. The nonlinear implementation is explicitly a kernel perceptron, not a kernel SVM solver.
- Chapter17:covariance eigenvalues about2.035532,.876431,.151866. One/two components retain.6644/.9504 variance and validation mean squared reconstruction norm1.1624/.3088. The denominator is rows, not rows×features.
- Chapter18:k2/k3 inertia74.8438/43.6011; EM summed log likelihood rises from−150.3323 to−142.591, weights about.7467/.2533. The report distinguishes density and membership from failure probabilities.
- Chapter19:six frozen pooled development costs[1.0000,1.0370,.9444,1.0000,.5556,1.3519] select depth2 tree. The previously measured fixed final comparison is preserved in the authored explanation:10/18 kNN,16/18 NB,10/18 selected tree, with five missing-temperature errors for the selected tree. It motivates an honest failure discussion, not data replacement or post-test model selection. Rechecking final outcomes is reserved for the explicit final route.

## Three interactive tools

`node labs/s03-tabular/verify-demos.cjs` runs the actual scoped browser scripts using a dependency-free DOM stub. These checks exercise event handlers and reset state instead of copying formulas into a disconnected test. They establish numerical/script behavior; root owns rendered-site/browser layout review.

| Home | Scope, accessibility and static alternative | Deterministic calculations checked |
| --- | --- | --- |
|12 `sampling-tool`|Fixed18 paired development predictions; choose row versus whole-machine resampling or odds confidence multiplier. Labelled native inputs/buttons, polite output, counted table, titled/described reliability SVG, text/noscript arithmetic; disabled irrelevant controls.|64-bit xorshift mirrors Rust, seed7 gives[.5,.8333], next draw changes seed and Reset restores. Calibration gives Brier.1859 and bin0 count5/mean.1431/frequency.2. Sharpening keeps accuracy.6667 while changing Brier.|
|13 `split-tool`|Five scoped select cases; membership/counts, shared machines, prediction availability and denominator are text. Completed kNN results are explicitly stored illustrations, not browser model fits.|All five cases:108/36/18 shared,96/48/0,72/54/18,48/18/0,48/18/0; measured accuracies26/36,33/48,36/54,12/18,18/18. Reset selects strict.|
|14 `geometry-tool`|Same named XOR corners/query across distance, threshold, kernel, projection and center-assignment views. Label/value bounds update by mode; scoped native controls, current-view SVG description, table values, exact static examples. Later chapters link here.|Horizontal scale1→2 changes tied squared distances2→5; threshold1→.9 sends query left→right; RBF gamma.5 gives exp(−1)=.3679; projection0°→45° changes lost norm1→0; center1 x0→1→2.5 changes distances2→1→3.25 and eventually assigns center2. Reset restores distance1.|

The symmetric query intentionally cannot show a neighbor-rank reversal under scaling; the lesson supplies a separate asymmetric worked example and the tool explicitly names its tie limitation. No widget claims to execute or verify learner Rust. No external scripts, chart dependencies or progress gates are used.

## Prerequisite and exposition audit

The section assumes everyday Rust collections, iterators, structs and Result handling, plus earlier-course prediction and loss distinctions. It does not assume prior statistics or classical ML:

- Chapter12 introduces population/sample/statistic, paired replacement draws, percentile indexing, independent units and confidence-procedure interpretation through explicit small counts before notation. Calibration, density/probability distinctions and Brier arithmetic are defined locally.
- Chapter13 works median/imputed mean/population variance and category bit layout before implementation. Feature availability precedes splitting, and four-entity/time examples separate group leakage from future leakage. Missing values are not zeros; unknown categories are not fitted categories.
- Chapter14 computes squared distance and a ranking reversal under units, then vote fractions and Gaussian log terms with priors/spreads. Conditional independence and Gaussian treatment of binary coordinates are named assumptions.
- Chapter15 works Gini-weighted splits, recursive partitions, actual resampling and residual updates. It distinguishes bagging from random feature selection and squared-error gradient boosting from logistic boosting.
- Chapter16 defines signed labels, affine score, margin, hinge and regularization before update rules; it separates the linear SVM objective from the nonlinear kernel-perceptron teaching branch.
- Chapter17 works an explicit3-row covariance matrix, its two eigenpairs, normalization/Rayleigh updates, projection/reconstruction shapes, sign ambiguity, orthogonality, condition number and eigengap sensitivity. It names covariance-formation and fixed-power limits rather than suggesting a general eigensolver.
- Chapter18 computes a Lloyd update and summed inertia, two-component posterior fractions, fractional counts/mean/variance, log normalization and negative-log-density examples. Label-free components, local optima, empty components, flooring and projection loss are explicit.
- Chapter19 computes asymmetric error cost before search, explains eligible-row group/time CV versus random folds, pooled denominators and selection bias, then works residuals and descriptive slices. Nested CV is substantively explained as an unimplemented extension; no topic is silently optional.

All chapter terms have canonical point-of-use glossary links and immediate explanations. Ordered step IDs match metadata exactly; original meaningful topic anchors remain where applicable or as nearby compatibility anchors. Source-tagged reference excerpts and original bibliographies remain, with a boundary explaining that their miniature code/results differ from the new shared lab. Existing source/evidence audit reports informed the section; no further literature delegation or fabricated source experiments were used.

## Validation and limits

Completed: `cargo fmt --check`, strict `cargo clippy --all-targets -- -D warnings`,9 passing unit tests,8 successful baseline runs,8 expected `GOAL_NOT_MET:` learner checks,8 successful solution runs and8 successful solution checks, plus the three actual JavaScript demo checks. CLI/runtime errors are not relabelled as unmet learning goals. Ch19 checks were audited to ensure no final outcomes are computed or printed. Scratch authoring generators and old output copies were removed; the useful dataset provenance generator remains.

Finite/shape guards are supplied. Exact checks cover discrete identities/counts; tiny numerical examples use declared absolute tolerances, including1e−8 eigenvalue/EM monotonicity,1e−10 unit norm and1e−14 squared eigenpair residual where appropriate. These checks establish implementation behavior on inspectable examples, not population coverage or general convergence. Public teaching model fields must retain their documented invariants when manually edited.

Residual limitations are intentional and taught: six independent machines for uncertainty; correlated/shifted sensors; small noisy labels; Gaussian approximation to binary inputs; inefficient full neighbor/kernel scans; small fixed forests; squared-loss rather than logistic boosting; kernel perceptron rather than dual SVM; covariance power iteration rather than robust SVD; diagonal GMMs and local optima; no nested CV; and descriptive final slices without causal identification. The final model is allowed to fail visibly on the fixed dataset.
