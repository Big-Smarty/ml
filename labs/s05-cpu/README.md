# One dense kernel, four CPU experiments

This independent, dependency-free Rust package carries the same `X[m,k]`, stored `W[n,k]`, bias `[n]`, and output `Y[m,n]` through measurement, GEMM traversal, threads/training, and SIMD. Stable Rust 1.89+ is required for the explicit AVX-512F extension. All defaults are tiny, deterministic, offline computations. Original `projects/ch25`–`ch28` remain preserved references.

```sh
cargo run --manifest-path labs/s05-cpu/Cargo.toml -- 25
cargo run --manifest-path labs/s05-cpu/Cargo.toml -- 25 --check
cargo run --manifest-path labs/s05-cpu/Cargo.toml -- 25 --solution --check
cargo run --release --manifest-path labs/s05-cpu/Cargo.toml -- 25 --solution --bench
```

Replace `25` with `26`, `27`, or `28`. From the course root, `just lab NN` and `just lab-check NN` run the learner version. `--check` cannot be combined with `--bench` or `--avx512`. Unknown flags are errors. Chapter 28 supports `--avx512` for explicit execution of the wide dot; unavailable hardware returns an unsupported error. No downloads, GPU, or paid compute are involved.

## Baselines, goals, and honest completion

| Chapter | Supplied working baseline | Learner implements | Delivered learner check |
|---|---|---|---|
| 25 | Correct dense result and one observable timed call | Complete warmup and repeated-measurement procedure | `GOAL_NOT_MET:` / exit 1: requested 14 real calls, observed 1 |
| 26 | Correct row-column-inner GEMM plus bias | Row-inner-column accumulation, clipped tile loops, reuse | `GOAL_REVIEW_REQUIRED:` / exit 3: one whole tile instead of required partition |
| 27 | Correct serial dense inference and actual scalar training | Scoped disjoint inference, private training shards, fixed reduction | `GOAL_REVIEW_REQUIRED:` / exit 3: serial calling-thread receipt |
| 28 | Correct scalar dense/dot | Full lane grouping, AVX2/FMA, guarded AVX-512F, tails | `GOAL_NOT_MET:` / exit 1: scalar total in lane zero rather than required lane sums |

Every `--solution --check` returns 0 after automatic numerical/behavioral checks. Chapters 26–28 retain a manual source-review status for the learner after their automatic checks pass: output equality, tile counters, worker IDs, and backend labels do not prove a particular implementation or mastery. Use the lesson and `lab.manual_checks` rubric. Numerical failures are not disguised as manual review; invalid inputs and runtime failures remain ordinary errors.

Learner files are `src/chNN.rs`; complete explained solutions are `src/solutions/chNN.rs`. Supplied `common.rs` owns the scalar dense oracle, validation, fixture generation, and benchmark formatting. `training.rs` adapts the preserved one-output scalar regression/gradient reference. Do not replace complete algorithms with changed labels or counters to satisfy checks. Inspect source and explain the invariant as well.

## Exact experiment controls

There is no arbitrary flag parser for model dimensions. The following named source values are the intended editable controls; generated benchmark shape labels use those values automatically. Change one control at a time and keep the previous result for comparison.

| Experiment | Exact location and editable value |
|---|---|
| Dense benchmark shape | `src/common.rs`, `benchmark_with`, local `s = Shape { m:37, k:65, n:31 }` |
| Warmups and repetitions | `src/common.rs`, the `measure(..., 3, 11)` calls in `benchmark_with`; also `src/ch26.rs::experiment` for prepared GEMM |
| GEMM shape and block | `src/ch26.rs::experiment`, local `s`, and `for block in [4,16,32]`; these feed both learner and solution prepared-layout timing |
| Dense inference workers | Last argument of `parallel_dense(...,3)` in `src/ch27.rs::dense` and `src/solutions/ch27.rs::dense`; update the corresponding run label too |
| Training workload | `src/ch27.rs::train`, tuple `(batch,features,threads,steps,rate) = (31,4,3,60,0.2)`; both learner and solution use it |
| SIMD fixture widths and lengths | `src/ch28.rs::verify`: widths `[4,8,16]`, lengths `(0..40).chain([65,1003])`, and last-element impulse lengths |
| Deterministic data variation | `common.rs::values(n,salt)` call sites in `benchmark_with`: salts 11 for X, 47 for W, 19 for bias. There is no hidden random seed. |

Copy the three-loop intermediate from your Chapter 26 work before adding tiles, or inspect the preserved `projects/ch26/src/main.rs` row-inner-column implementation. `common::dense` remains a stable scalar checkpoint throughout. The final SIMD and threaded wrappers target the same dense operation; they are separately measured variants, not a claim that all optimizations necessarily compose into the fastest kernel.

## Numerical and timing boundaries

The hand result is `[22,28,49,64]`. Dense checks cover nonsquare shapes, nonzero biases, odd lengths, and twice-used output buffers. Dense dimensions must be positive and products must not overflow; nonfinite inputs/parameters and nonfinite computed outputs are errors. Empty equal-length dots return zero. Empty scalar prediction returns an empty vector; empty training is rejected.

For bounded fixtures, f32-vs-f64 comparison uses `2e-6 * max(k,1) + 2e-5 * abs(reference)` per output. This accounts for reduction grouping and FMA, with impulse tests to catch lost terms. It is not a universal precision guarantee. Training compares every mean loss/gradient with the scalar calculation using `1e-6 + 1e-5 * abs(reference)`, plus hand derivatives and an atomic-overflow update check. The 31-row training loss must fall by at least 1000× after 60 steps; this verifies mechanics, not generalization.

Use release builds for performance; `--bench` rejects debug builds. Timings have three warmups and eleven samples, observable outputs, median and full range, and an f64 check before measurement. The scalar and prepared GEMM arithmetic calls allocate no buffers in the timed region. Prepared GEMM includes validation/bias initialization but excludes the W-to-B transpose. Threaded call timing includes worker launch/join and handle/receipt allocations and is labeled call-level latency, not allocation-free kernel latency. Checked SIMD includes per-dot validation and dispatch. A separate dense interval includes output allocation, call, and drop with inputs already resident. No interval includes printing or oracle work. Cold-array bytes are an estimate, not measured DRAM traffic.

Record `rustc -Vv`, CPU model, OS, build flags, precision, shape, threads/block, timed boundary, warmups, samples, median, range, and checksum. No algorithm has a promised speedup. The small thread/SIMD wrappers can lose to scalar overhead. Assembly inspection is separate evidence for auto-vectorization.

## Checks

```sh
cargo fmt --manifest-path labs/s05-cpu/Cargo.toml --check
cargo clippy --manifest-path labs/s05-cpu/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path labs/s05-cpu/Cargo.toml
node labs/s05-cpu/check_demos.cjs
```

The browser tools live only in chapters 25 and 27. Their deterministic pure calculations are checked by the last command. They illustrate index coverage, arithmetic, and rounding; they perform no real CPU feature detection, cache simulation, or timing. All narrative results remain available without JavaScript.
