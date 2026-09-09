# Classical learning on the maintenance table

This independent, offline Rust package is the working project for Chapters 12–19. It requires stable Rust and uses only the standard library. Each learner module starts with a useful bounded algorithm; running it successfully is a starting checkpoint, not completion of the chapter goal.

From the repository root:

```bash
just lab 12
just lab-check 12
cargo run --manifest-path labs/s03-tabular/Cargo.toml -- 12 --solution
cargo run --manifest-path labs/s03-tabular/Cargo.toml -- 12 --solution --check
cargo test --manifest-path labs/s03-tabular/Cargo.toml
```

Replace `12` with any chapter through `19`. `--check` calls the actual selected implementation on changed data. Intentional unmet comparisons print `GOAL_NOT_MET:` and exit 1; input/runtime errors remain ordinary errors. Completed solutions pass. Unknown arguments are rejected. There is no watch runner, download, hidden training service or required hardware.

| Chapter | Working learner checkpoint | Substantial learner implementation |
| --- | --- | --- |
| 12 | Row-normal accuracy interval; one global calibration bin | Whole-machine bootstrap and nonempty reliability bins |
| 13 | Mean imputation in original units; one row holdout | Training medians, centering/scaling, vocabulary and four split policies |
| 14 | One nearest neighbor; equal-prior unit-variance class centroids | General k-neighbor voting; fitted class priors and variances |
| 15 | Median stump; cloned trees; global-mean regression | Recursive CART, resampled bagging, random-feature forests, residual boosting |
| 16 | Perceptron update; two fixed RBF prototypes | Regularized hinge SVM update; learned kernel-perceptron coefficients |
| 17 | High-variance coordinate axes with diagonal covariance | Full covariance and orthogonal power-iteration components |
| 18 | Fixed-center assignment; equal-weight unit-variance mixture | Lloyd mean updates and stable E/M updates with learned spreads |
| 19 | One legal holdout and first candidate only | Three group/time folds and all six declared candidates |

Edit `src/chNN.rs`; inspect the separate explained `src/solutions/chNN.rs` after an attempt. Later chapters import completed earlier checkpoints so an unfinished earlier goal does not block the next session. Once your version passes, substitute the corresponding learner import if you want a complete personal implementation chain. Shared CSV parsing, random draws, finite/shape validation, transformations and scoring remain supplied. Public model fields make arithmetic inspectable; manual mutation of a fitted model must preserve its documented shape/finite invariants.

## Dataset provenance and prediction boundary

`data/maintenance.csv` is **course-authored synthetic data**, not a downloaded or real plant dataset. It contains 24 machines × 12 days = 288 rows. `data/generate.py` discloses its Python-standard-library generation procedure with `random.Random(20260909)`. The delivered CSV is frozen, with FNV-1a64 `6dccde7b74debd05`. The hash is an identity check, not a cryptographic integrity claim. Regenerate only to establish a deliberately new dataset version and a new evaluation cycle; do not regenerate or tune it after seeing final results.

A shared latent wear process produces correlated temperature/vibration measurements and repeated-machine dependence. Outcomes are Bernoulli draws, creating irreducible overlap. Some readings are missing, and missingness partly depends on the outcome process. Days 9 onward introduce a +5 temperature measurement bias and a shifted outcome tendency. Some final machines enter a previously unseen `turbo` regime. These are disclosed teaching mechanisms, not claims that the same mechanisms explain every observed error.

| Column | Meaning and availability |
| --- | --- |
| `id` | Stable `machine * 12 + day` identity; split/audit metadata |
| `machine`, `day` | Repeated unit and prediction day; excluded from learned features |
| `temperature`, `vibration`, `load` | Three numeric measurements available at prediction, blanks permitted |
| `regime` | Category available at prediction |
| `failure` | Binary next-day outcome, available at day + 1 |
| `repair_after` | Post-event field available at day + 1; explicit forbidden leakage trap |
| `probability` | Frozen sensor heuristic available at prediction; Chapter 12 only, excluded from learned-model features |

A completed transform stores training medians, imputed means, population scales and sorted category vocabulary. The regular representation has width 9: three standardized sensors, three missingness indicators, two known-category indicators, one unknown-category indicator. A constant numeric feature uses scale 1. An entirely missing training numeric column returns an error. Held-only categories never change fitted width. Chapters 17–18 use just the first three numeric coordinates, then two fitted PCA coordinates for clustering.

## Frozen evaluation protocol

The development pool is machines 0–17 on days 0–7 (144 rows). Chapters 13–18 normally train machines 0–11 through day 3 (48 rows) and validate machines 12–17 on days 5–7 (18 rows); day 4 is a gap. Chapter 12 evaluates the frozen heuristic on those same 18 development validation rows.

Chapter 19 rotates three blocks of six development machines. Each fold trains earlier rows from the other two blocks and validates days 5–7 of the held block. There are 54 distinct eligible validation predictions per candidate; early/gap rows are not ordinary random-fold validation observations. Every fold fits preprocessing from its own raw training rows. Stable ID sorting fixes traversal and tie behavior.

The candidate order is kNN k3/k7, Gaussian NB variance floor .02/.2, tree maximum depth 2/4. Selection minimizes pooled `(4*false_negatives + false_positives)/n` at threshold .5, retaining the earliest candidate on ties. The candidate budget, cost ratio and threshold are frozen exercise choices. Brier and accuracy are accompanying diagnostics. SVM hard decisions are explicitly not calibrated probabilities; tree boosting here fits squared residuals, not logistic gradients. The nonlinear Chapter 16 model is a kernel perceptron, not a kernel SVM dual optimizer.

Ordinary Chapter 19 runs and checks do not evaluate or print final outcomes. After recording the completed protocol and selected configuration, explicitly open the final report:

```bash
cargo run --manifest-path labs/s03-tabular/Cargo.toml -- 19 --final
# Or inspect the completed checkpoint:
cargo run --manifest-path labs/s03-tabular/Cargo.toml -- 19 --solution --final
```

Final models refit on all 144 development rows through day 7; day 8 is a gap. The final partition is machines 18–23 on days 9–11 (18 rows). The report compares one development-selected finalist from each of three families, training-majority and training-prevalence baselines, whole-machine bootstrap intervals, residual rows and missingness/regime slices. Comparing final numbers does not permit choosing a new winner on that same test. Only six final machines make uncertainty and slice conclusions fragile. The CSV is source-visible for transparency; `--final` is a pedagogical evaluation boundary, not access control.

## Controlled variations and numerical checks

The lessons identify exact function names, source values or local cloned inputs for every variation. The CLI supports chapter number, `--check`, `--solution`, and Chapter 19 `--final`; it does not pretend to accept extra experimental flags. Examples include changing the report's NB variance floor, boosting rounds, kernel gamma or PCA iteration count, removing the same sensor column from both prepared sides, and adding a report-only final-day slice. Keep the canonical CSV, final partition and capstone candidate grid unchanged.

Numerical checks use declared absolute tolerances appropriate to tiny deterministic examples: exact counts/membership where discrete, roughly 1e-12 for simple normalized arithmetic, 1e-8 for the known PCA eigenvalue and EM likelihood monotonicity, and a squared eigenpair residual below 1e-14. The implementation is teaching-scale dense arithmetic, not a general robust eigensolver or a scalable tree/kernel library. Fixed iteration counts do not establish convergence on every input.

The three browser illustrations use scoped vanilla JavaScript and labelled native controls. They illustrate calculations; they do not execute Rust. Their fixed cases, changed inputs and resets can be verified without dependencies:

```bash
node labs/s03-tabular/verify-demos.cjs
cargo fmt --manifest-path labs/s03-tabular/Cargo.toml --check
cargo clippy --manifest-path labs/s03-tabular/Cargo.toml --all-targets -- -D warnings
```

Original `projects/ch12` through `projects/ch19` remain preserved references with their own miniature fixtures and assumptions. The chapter source-tagged excerpts label that boundary explicitly.
