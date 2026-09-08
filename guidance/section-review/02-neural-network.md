# Section 02 — Build a neural network (chapters 07–11)

Reviewed 2026-09-08 against base commit `43a16cc2c0b09a2bc4c75f79869cc2b9ab4fca5c`. Complete and frozen; one section reviewer, no nested agents.

## Coverage

Read every `lesson.html`, `meta.json`, and `research.md` in `chapters/{07,08,09,10,11}`; every reference and starter `src/main.rs` and `Cargo.toml` in `projects/ch{07,08,09,10,11}`; and both exercise/solution `chNN_01.rs` files for all five chapters. Read the section's `course.json` assignments, shared consistency/numerics/authoring contracts, chapter template, and applicable Rust review skills. Checked the Chapter 6 prerequisite and Chapter 12 continuation interfaces in their lessons.

- **07:** XOR representational argument, all nine backpropagation derivatives, unchanged forward-pass weights, stable BCE, mean reduction, worked forward/backward values, starter derivative targets, and sigmoid/tanh exercise mapping.
- **08:** graph identity versus value equality, shared operand accumulation, reverse topological order, adjoint clearing, local derivatives, nonfinite handling, scalar-engine limitations, and starter/solution mappings.
- **09:** nonsquare row-major layouts; all forward, input-gradient, weight-gradient, and bias-gradient arithmetic; supplied upstream reduction; shape validation; finite-difference scalar objective; and forward/offset exercises.
- **10:** IDX structure and bounds, normalization, ten-class logits/softmax/cross-entropy, minibatch mean including the last batch, held-out metrics and split policy, CLI examples, and parser/softmax/log-sum-exp exercises.
- **11:** parameter packing, ReLU derivatives and initialization variance, mean loss/gradient, momentum and Adam equations, checkpoint parsing and write path, exact-resume assumptions, evaluation-only CLI, and working SGD starter/momentum exercises.

## Material findings and fixes

1. **Checkpoint saving could destroy an unrelated sibling file.** `save(model.bin, ...)` previously opened `model.tmp` with truncation, then renamed it away. A checkpoint target ending in `.tmp` also acted as its own temporary file. Changed only the temporary-path construction and opening: append `.tmp` to the complete path and use `File::create_new`. A collision now returns an error identifying its path without truncating or removing either existing file. Serialization fields and public/CLI interfaces are unchanged. The lesson explains collision behavior.
   - Added `checkpoint_save_preserves_temporary_siblings`. Before the implementation change, it failed because `model.tmp` had disappeared after save. After the fix, it checks an unrelated sibling survives, an existing appended temporary file blocks replacement while preserving both byte sequences, and a `checkpoint.tmp` target saves/restores successfully.
2. **The Chapter 10 exercise description contradicted the stable fused loss calculation.** It instructed readers to subtract the target logit from the standalone log-sum-exp result. At ten equal logits of `1e16`, that separately rounded calculation gives `2` instead of `ln(10)`. The exercise now distinguishes its valid standalone primitive from the cancellation-preserving cross-entropy ordering already used by the reference and tested by `cross_entropy_preserves_shared_offsets_and_parser_contract`. No exercise implementation or intended TODO changed.
3. **Exact resume needed its data-order boundary stated.** The Chapter 11 lesson now states that CLI checkpoints are saved at epoch boundaries, use sequential batches of 32 and a fixed learning rate, and require the same training rows/order for exact continuation. This explains why this particular implementation needs no shuffle, minibatch cursor, or scheduler state. No checkpoint format change was needed.

No other material correction was found. Existing measured MNIST evidence remains unchanged.

## Validation

For each `NN` in `07 08 09 10 11`, ran these commands for both `projects/chNN/Cargo.toml` and `projects/chNN/starter/Cargo.toml`:

```sh
cargo fmt --manifest-path MANIFEST -- --check
cargo clippy --offline --manifest-path MANIFEST --all-targets -- -D warnings
cargo test --offline --manifest-path MANIFEST
cargo run --offline --quiet --manifest-path MANIFEST
```

All ten format and strict-Clippy checks passed. Reference test counts: **07: 4, 08: 4, 09: 3, 10: 4, 11: 8**, all passed (23 total). Every starter ran successfully; each starter test returned exit 101 only at its one intended unfinished operation. The new checkpoint regression was demonstrated failing before the fix with:

```sh
cargo test --offline --manifest-path projects/ch11/Cargo.toml checkpoint_save_preserves_temporary_siblings
```

For every exercise and solution, ran `rustc --edition=2021 --test exercises/cpu/KIND/chNN_01.rs -o /tmp/section02-KIND-N`, followed by the resulting executable. All ten compiled; all five solution tests passed; all five exercise tests failed at their intended TODO with exit 101.

Bounded default outputs agree with the teaching claims: XOR mean BCE **0.001962** and all classes correct; dense arrays match the worked arithmetic; linear fixture held-out loss **2.3026 → 0.0133**, accuracy **100%** after training; MLP fixture **2.6914 → 0.0029**, accuracy **100%**, restored optimizer step **300**; working SGD starter **2.7378 → 0.3333**.

A section-scoped HTMLParser check extracted all eight `data-source` code excerpts, normalized whitespace, and verified each against its actual source file: all matched. Detailed command output is in `/tmp/section02-checks.log` for this session.

## Limits and cross-section status

No extended MNIST training, downloads, GPU runs, site build, global verifier, or new external-source review was performed. Prior source verification and measured MNIST results were preserved; this review checked local mathematical and implementation agreement. The safe writer still does not sync the containing directory, as already documented; an existing temporary file requires inspection before retrying. No shared files, other chapters, learner work, existing review logs, validation JSON, dataset files, or retained checkpoints were modified. No cross-section interface change or outstanding cross-section issue.
