# Section 05 review — Make the CPU faster

Reviewed 2026-09-08 by the assigned GPT-6 Astra Medium section reviewer. Status: complete and frozen.

## Coverage and outcome

Read all four lessons, metadata, research notes, reference `src/main.rs` files, starter sources and manifests, and Rustlings exercise/solution pairs for chapters 25–28. Checked the section's `course.json` topic assignments and `exercises/cpu/info.toml` mappings. Read the shared consistency, numerical, and authoring contracts and applicable Rust review skills. Checked the relevant Chapter 24 score/dot-product transition and Chapter 29 CPU-to-GPU continuity statements.

No material correctness, consistency, or understandability defect found; no lesson or implementation changes needed.

- **25:** Verified the dense layout `[batch,in_features] × [out_features,in_features]ᵀ + bias`, hand result −0.65, 131,072-FLOP example, 72-byte/24-FLOP example, tolerance comparison, preallocated timing boundaries, and qualified cold-array traffic estimate.
- **26:** Verified dense-weight transpose into GEMM B, all three loop orders, rectangular worked result `[22,28,49,64]`, one output reset, partial tiles, and saturating bounds. The source preserves increasing reduction-index order for each output.
- **27:** Verified one-output dense specialization, disjoint inference slices, immutable model use during gradient calculation, unequal-shard sums followed by one full-batch division, fixed handle reduction order, nonzero hand loss/gradient/update, and rejection before an overflowing update mutates the model.
- **28:** Verified length validation before unsafe calls, architecture gates, AVX2/FMA and AVX-512F runtime guards, unaligned in-bounds loads/stores, scalar tails, f64 oracle comparisons, dispatch-inclusive timing, and honest auto-vectorization wording.

All 12 tagged lesson excerpts match their declared source after HTML decoding and whitespace normalization (chapter counts: 2, 3, 4, 3). Exercise and starter TODOs are intentional and preserved.

## Validation

For each `NN` in `25 26 27 28`, ran:

```sh
cargo fmt --manifest-path projects/chNN/Cargo.toml --check
cargo clippy --offline --manifest-path projects/chNN/Cargo.toml --all-targets -- -D warnings
cargo test --offline --manifest-path projects/chNN/Cargo.toml
cargo run --offline --release --manifest-path projects/chNN/Cargo.toml
cargo fmt --manifest-path projects/chNN/starter/Cargo.toml --check
cargo clippy --offline --manifest-path projects/chNN/starter/Cargo.toml --all-targets -- -D warnings
cargo run --offline --manifest-path projects/chNN/starter/Cargo.toml
cargo test --offline --manifest-path projects/chNN/starter/Cargo.toml
```

All formatting, Clippy, reference tests, reference demos, and starter launches passed. Reference test counts were 3, 3, 6, and 5 respectively. Each starter test command failed only at its intended TODO; Chapter 28's supplied unequal-length check passed. The Chapter 27 default demo reduced MSE from 0.596916 to 0.001287. Chapter 28's default selected `avx2+fma` on this x86_64 Linux host.

Compiled each exercise and solution with `rustc --edition=2021 --test exercises/cpu/{exercises,solutions}/chNN_01.rs -o /tmp/section05-{exercises,solutions}-NN` and ran the resulting binary. All compiled; each solution passed its test, and each unfinished exercise failed at its intended TODO. Raw command output is in `/tmp/section05-checks.log` and `/tmp/section05-learning-checks.log`.

## Limits and cross-section interfaces

No shared interfaces changed and no outstanding cross-section issue found. No global build/verifier, extended workload, profiling counters, target cross-compilation, or separate AVX-512 CLI run was performed. Existing runtime-guarded tests exercise supported backends; they do not establish behavior on a different CPU or non-x86 target. Benchmark timings obtained during concurrent review are correctness smoke checks, not performance evidence. Root may run `cargo run --offline --release --manifest-path projects/ch28/Cargo.toml -- --avx512` if an explicit hardware-mode smoke check is desired.
