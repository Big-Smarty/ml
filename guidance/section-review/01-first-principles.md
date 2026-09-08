# Section 01 review: First principles (Chapters 01–06)

Reviewed 2026-09-08 by the assigned GPT-6 Astra Medium section reviewer against base `43a16cc2c0b09a2bc4c75f79869cc2b9ab4fca5c`.

## Coverage

Read every `chapters/01`–`chapters/06` lesson, metadata file, and research note; Chapter 1's `demo.js`; all six reference and starter `src/main.rs` files and Cargo manifests; and all twelve `exercises/cpu/{exercises,solutions}/ch01_01.rs`–`ch06_01.rs` files. Checked the corresponding course topics and Rustlings/Cargo mappings. Read the consistency, numerical, and authoring contracts and applicable Rust skills. Inspected Chapter 7's opening, continuity, derivative explanations, and reference forward/loss interface for the next-section transition.

The section develops scalar numerical gradients into analytical MSE gradients, vector regression and training-only scaling, binary likelihood and stable BCE, held-out evaluation, and minibatch/L2 optimization. Checked the worked arithmetic, factor-two and mean reductions, simultaneous updates, array order, gradient/model distinction, probability/class distinction, threshold selection, and unregularized curve reporting against their implementations. The explanations and exercises support the transition to Chapter 7's cached forward pass and manual chain rule.

## Findings and changes

No material correctness, understandability, or consistency issue required a change. Only this review report was added. Learner work in the Chapter 1 starter/exercise and Chapter 2 helper experiment remains untouched, as do existing review and validation records.

## Validation

For each `NN` in `01 02 03 04 05 06`, ran:

- `cargo fmt --manifest-path projects/chNN/Cargo.toml --check`: all pass.
- `cargo clippy --offline --manifest-path projects/chNN/Cargo.toml --all-targets -- -D warnings`: all pass.
- `cargo test --offline --manifest-path projects/chNN/Cargo.toml`: all pass, 17 reference tests total.
- `cargo run --offline --release --manifest-path projects/chNN/Cargo.toml`: all pass.
- The same formatting, Clippy, and test commands for `projects/chNN/starter/Cargo.toml`, plus `cargo run --offline --manifest-path projects/chNN/starter/Cargo.toml`: formatting, Clippy, and launch all pass. Chapter 1's three completed tests pass; Chapters 2–6 each fail exactly their one intentional TODO test (exit 101).
- `cargo test --offline --manifest-path exercises/cpu/Cargo.toml --bin chNN_01` and the corresponding `--bin chNN_01_sol`: all six solutions and the completed Chapter 1 exercise pass; Chapters 2–6 exercises each fail at the intended TODO (exit 101).

Imported only the existing consistency parser and whitespace-normalization helper, without running its global checker, to verify all 19 tagged source excerpts in the six lessons and all six continuity/conventions links. All pass.

Observed reference outputs agree with the lessons: Chapter 1 initial MSE 9 and learned line (2,1); Chapter 2 analytical/numerical gradient (−8,−2); Chapter 3 MSE 2.0695 and prediction 242.57; Chapter 4 BCE 0.009286; Chapter 5 threshold 0.60 and confusion counts (TP 1, FP 1, TN 7, FN 1); Chapter 6 final validation MSE 1.347 / 0.025 / 0.007 for slow / steady / L2.

Temporary command transcripts: `/tmp/ml-section01-reference-gates.json`, `/tmp/ml-section01-starter-gates.json`, and `/tmp/ml-section01-exercise-gates.json`.

## Limits and handoff

This was a source and scoped executable review. The browser demo was read and checked against the numerical algorithm, but no browser rendering or global site checks were run. No external APIs, GPU checks, long training, or new source claims were needed. Existing fixed-fixture and fixed-minibatch limitations remain accurately stated. No remaining cross-section issue was identified.

**Status: complete and frozen.**
