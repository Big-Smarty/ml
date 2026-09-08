# Section 04 review: Broader deep learning

Reviewed 2026-09-08 by the assigned single section reviewer against base `43a16cc2c0b09a2bc4c75f79869cc2b9ab4fca5c`. Status: complete and frozen.

## Coverage and findings

Read the complete `lesson.html`, `meta.json`, and `research.md` for chapters 20–24; each chapter's reference and starter `src/main.rs` and Cargo manifests; and all five matching `exercises/cpu/exercises/chNN_01.rs` / `solutions/chNN_01.rs` files. Checked their `info.toml` mappings and `course.json` topics. Applied the consistency, numerics, authoring, and Rust review contracts. Inspected the relevant Chapter 11 dataset/dense/classification interfaces, Chapter 17 reconstruction reduction, Chapter 19 evaluation handoff, and Chapter 25 introduction and dense-layout handoff.

No material correctness, consistency, or understandability issue found. No lecture or implementation changes were necessary.

- **20:** Traced padded cross-correlation, filter-plane indexing, max-pool routing, shared-weight gradients, stable cross-entropy, per-image updates, IDX validation, and fixture/MNIST distinction. Shape and hand arithmetic agree with code.
- **21:** Traced both residual blocks, ReLU gating of the shortcut, normalization statistics over 36 pixels, epsilon-aware normalization backward, and gradients through the second block to the first. The lesson explicitly distinguishes per-image normalization from batch normalization and source-related validation perturbations from independent examples.
- **22:** Checked autoencoder coordinate MSE and old-decoder gradient use, shared contrastive gradients through both views, epsilon-aware vector normalization, temperature factors, and anchor-mean InfoNCE. Training-pair retrieval is correctly separated from transfer claims and Chapter 17's row-mean squared reconstruction norm.
- **23:** Checked RNN/LSTM BPTT, shared parameter accumulation, full-sequence mean MSE, the 80/40 target boundary, state reset, teacher-forced evaluation, and the persistence comparison. Modern forget-gate attribution and the unfavorable LSTM baseline result remain explicit.
- **24:** Checked dot-product lookup semantics, simultaneous three-vector pairwise updates, per-triple regularization, held-out interaction exclusion, candidate sorting, and the distinct Recall/Precision/nDCG formulas. One-positive Recall equals hit rate only under the stated fixture condition.

## Validation

For every `NN` in `20 21 22 23 24`, ran:

```sh
cargo fmt --check --manifest-path projects/chNN/Cargo.toml
cargo clippy --offline --all-targets --manifest-path projects/chNN/Cargo.toml -- -D warnings
cargo test --offline --manifest-path projects/chNN/Cargo.toml
cargo run --offline --release --manifest-path projects/chNN/Cargo.toml
```

All passed. Reference test counts: 20: **6**, 21: **4**, 22: **8**, 23: **5**, 24: **4** (27 total). All five release demos reproduced the lesson's printed values and ranking lists at the displayed precision.

For each starter, ran `cargo fmt --check`, `cargo clippy --offline --all-targets -- -D warnings`, `cargo run --offline`, and `cargo test --offline`, with `--manifest-path projects/chNN/starter/Cargo.toml`. Formatting, Clippy, and launches passed. Each starter test exited 101 solely at its intentional `todo!` (one test per starter).

Compiled each corresponding exercise and solution with `rustc --edition=2021 --test FILE -o /tmp/ml-section04-checks/BINARY`, then ran it. All five solutions passed their single test; all five exercises compiled and failed at their intended `todo!`. No learner work was changed.

A section-scoped HTML parser extracted all `data-source` code blocks and checked whitespace-normalized containment in their named source files: **14/14 matched** (3, 4, 3, 3, 1 by chapter). Reviewed untagged exercise expressions and numerical examples directly against the code. Temporary command logs are in `/tmp/ml-section04-checks/`.

## Limits and cross-section issues

No outstanding cross-section interface issue. No interfaces or shared files changed. No new external research was required to resolve a disputed claim. Prior source-verification and MNIST evidence remain intact; this review did not rerun MNIST, access the official test set, perform extended training, or run hardware/global site checks. The small deterministic demonstrations establish the documented mechanisms, not generalization or comparative performance beyond their stated fixtures.

Only changed path: `guidance/section-review/04-deep-learning.md`.
