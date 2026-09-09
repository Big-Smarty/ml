# Section 05 — Make the CPU faster

Completed 2026-09-09 by the single GPT-6 Astra High section owner. Scope: chapters 25–28, independent `labs/s05-cpu`, and this report. No authorship delegation, changes to preserved original projects, commits, pushes, or publication. Active authoring, chapter, assignment, numerical and updated delivery-status contracts applied. Rust skills applied to validation, scoped joins, architecture gates and unsafe invariants.

The project carries one dense operation, X[m,k] × W[n,k]ᵀ + bias[n] → Y[m,n], through distinct measured implementations. Prepared GEMM explicitly stores B=Wᵀ as [k,n]. Threaded training uses the one-output instance of the same operation. The final SIMD wrapper integrates checked dots into the original dense layout. These are preserved comparison checkpoints, not a promise that every optimization combines profitably.

## Four-chapter design matrix

| Chapter / original topics (verbatim) | Objectives and ordered explanation mapping | Meaningful learner work / working baseline | Checks, transfer and sessions | Prerequisite audit |
|---|---|---|---|---|
| **25** — Floating point; profiling; memory costs; arithmetic intensity | `floating-point`: hand dense answer, shapes, f32/f64 and rounding. `measurement`: complete warmups/samples and timing boundary. `work`: median/range, FLOPs, throughput and configuration. `memory`: array traffic, hierarchy, intensity and focused trace. `practice`: profile and change one workload. `review`: evidence record. | Baseline produces [22,28,49,64] and times one real dense call. Learner implements the entire repeated measurement procedure with propagated errors, not a median-only expression. | Callback counts check 3+11, 2+5, 0+1; only measured calls become samples; invalid plans/errors fail. Dense hand result plus varied-shape f64 oracle. Transfer: change m and compare call versus output-allocation-inclusive intervals. **2 sessions: 36,36 minutes.** | Retrieves dense row products from 09/24 without assuming matrix/GEMM expertise. Defines m,k,n, storage and operations before equations. Introduces floating-point precision and measurement terminology locally. |
| **26** — Loop ordering; cache locality; blocking; allocation reuse | `formula`: concrete W-to-B transpose and offsets. `ordering`: accumulate [1,2]→[7,10]→[22,28]. `allocation`: reused output and nonzero bias. `blocking`: clipped tiles on 3×5×7. `practice`: implement and measure tiles. `review`: bounded interpretation. | Correct row-column-inner dense/GEMM baseline. Learner first writes full row-inner-column accumulation, then three clipped tile loops, initialization and actual tile counting. Separate full solution and original row-inner-column reference are available. | Nonsquare shapes, blocks 1/4/16/usize::MAX, bias, repeated output reuse, zero-block rejection, f64. Structural source review remains required. Transfer: block 2 on 3×5×7 visits 24 tiles; measure shape/block changes through named controls. **2 sessions: 35,37 minutes.** | Recalls timing and row-major layout; works numerical products and offsets before tiling. Explicitly explains the changed B layout and transpose cost. Does not assume cache expertise. |
| **27** — Partitioning; scoped threads; gradient reduction; reproducibility | `partition`, `inference-checkpoint`, `partition-transfer`: independent output ownership. `gradients`, `gradient-checkpoint`, `reproducibility`: private derivative sums, full-batch normalization, fixed grouping. `run`, `practice`, `review`: real simultaneous training and launch-inclusive policy measurement. | Correct serial dense inference and actual scalar training run initially. Learner replaces complete inference execution with scoped workers and implements full private-shard training orchestration and reduction. Incidental model/fixture/scalar derivative code stays supplied. | Worker receipts plus dense parity for unequal/short/more-workers-than-rows cases. Mean loss and every gradient compare with scalar on 31/17/2 rows. Hand gradient/update, atomic overflow rejection, real 60-step loss decrease. Source rubric covers ownership/joins/reduction. Transfer: implement a threshold wrapper and compare unfamiliar shape. **3 sessions: 37,39,36 minutes.** | Retrieves one-neuron MSE derivatives from earlier course, then derives the two-row example explicitly. Defines Model versus Gradient. Explains borrowing/scopes before code. No prior thread-performance knowledge assumed. |
| **28** — Auto-vectorization; AVX2 and FMA; feature detection; tails; AVX-512 extension | `lanes`, `automatic`, `lane-checkpoint`: arithmetic grouping and portable algorithm. `dispatch`, `practice`, `safe-transfer`: full AVX2/FMA kernel and proved caller boundary. `avx512`, `simd-measure`, `review`: complete wide extension and actual selected backend evidence. | Correct scalar dot/dense baseline; scalar total initially occupies lane zero. Learner implements full lane-grouped accumulation plus tail, then actual guarded AVX2/FMA and AVX-512F loops, reductions and dispatch. Separate solution integrates SIMD in dense inference. | Individual lane values, empty/short/odd lengths 0..39,65,1003, final-element impulses around vector boundaries, scalar/f64 and dense agreement. All supported explicit backends execute locally; source safety review remains separate. Transfer: nonzero tail/bias and a wide dense wrapper; no hardware claim when unsupported. **3 sessions: 39,38,37 minutes.** | Retrieves the same dense row dot and rounding example. Teaches lanes, prefixes, tails, horizontal reduction and FMA arithmetically before intrinsics. Architecture-specific execution is gated, while the portable algorithm remains runnable everywhere. |

All original meaningful section anchors are retained. Every topic key exactly matches `course.json`; step arrays match HTML order; every session has implementation and transfer work. Each step begins with h2 and an explicit goal/question. Exact Rust excerpts are tagged with existing `data-source` paths and checked against source contents. Each chapter has immediate glossary definitions with canonical links, progressive hints, explained answers, learner entry/check metadata and a separate completed solution.

## Automatic evidence and manual review

The final package has seven passing Rust tests. All four ordinary baseline commands exit 0. Delivered learner check statuses are explicit:

| Chapter | Learner `--check` | Separate solution `--check` |
|---|---|---|
| 25 | Exit **1**, `GOAL_NOT_MET:` — 14 requested calls versus one actual call | Exit **0** |
| 26 | Exit **3**, `GOAL_REVIEW_REQUIRED:` — correct numeric baseline, one whole tile rather than requested clipped partition | Exit **0** |
| 27 | Exit **3**, `GOAL_REVIEW_REQUIRED:` — correct dense baseline, calling-thread receipt rather than distinct workers | Exit **0** |
| 28 | Exit **1**, `GOAL_NOT_MET:` — expected individual lane sums and tail differ from scalar-total baseline | Exit **0** |

After successful automatic checks, learner chapters 26–28 still return the explicit manual-review status. Metadata `lab.manual_checks` and lesson rubrics cover loop structure, contiguous access, disjoint ownership, private gradients, reduction order, actual intrinsics and SAFETY proofs. Labels/counters/receipts are supplemental evidence, not proof of mastery. Solution source was reviewed against those rubrics: three clipped tile loops and one bias initialization; nonempty paired disjoint chunks; immutable parameters; scoped launch/join errors; private loss/gradient sums reduced in creation order and averaged once; cfg-gated actual AVX2/FMA and AVX-512F loops with checked feature/bounds/store preconditions and scalar suffixes. Timing/causal interpretation remains a human review task.

Validation executed:

- `cargo fmt --manifest-path labs/s05-cpu/Cargo.toml --check` — pass.
- `cargo clippy --manifest-path labs/s05-cpu/Cargo.toml --all-targets -- -D warnings` — pass, no blanket lint suppression.
- `cargo test --manifest-path labs/s05-cpu/Cargo.toml` — 7 passed, 0 failed.
- Every baseline, every separate solution check, and all four expected learner-check statuses verified by actual process exit/output.
- Every solution `--bench` executed in release; explicit `28 --solution --avx512` executed and returned backend `avx512f`, dot 9.5 for 19 half-scale ones.
- Numerical tool command `node labs/s05-cpu/check_demos.cjs` — pass.
- Scoped HTML/metadata check: unique IDs; exact ordered step arrays; exact topic keys/actual step targets; 30–45 minute session totals; all original section anchors; exact source snippets; existing source links; no old starter/Rustlings workflow — pass for all four chapters.
- `cargo miri --version` found Miri unavailable on the installed stable toolchain. No Miri result is claimed; no toolchain was installed. Unsafe code was source-reviewed and exercised on actual supported x86 hardware. Non-x86 compilation/execution was not tested here.

Numerical dense tolerance is `2e-6*max(k,1) + 2e-5*abs(reference)` for these bounded fixtures, with separate exact impulse tests that catch dropped work. Gradient tolerance is `1e-6 + 1e-5*abs(reference)`, plus independent hand arithmetic. Invalid shapes, dimension-product overflow, nonfinite input/model values, output overflow, zero/excessive thread requests and empty training are rejected. An overflowing proposed parameter update leaves the original model unchanged. The 31×4 training fixture with 3 requested workers, 60 steps and learning rate 0.2 measured MSE **0.611521 → 0.000080**; this is synthetic training mechanics, not generalization.

## Actual local measurements

Measured 2026-09-09 on **AMD Ryzen 9 9900X, 12 cores/24 logical CPUs**, x86_64 Linux, `rustc 1.96.0 (ac68faa20 2026-05-25)`, LLVM 22.1.2, default Cargo release profile. `RUSTFLAGS` and `CARGO_ENCODED_RUSTFLAGS` were unset. No affinity, isolated core, fixed frequency, custom target-cpu flag or hardware counters were configured. This was a shared working machine, so these are local observations, not stable product benchmarks.

All rows below use deterministic X=[37,65], W=[31,65] (prepared B=[65,31] for GEMM), f32, **3 warmups + 11 samples**, 149110 conventional FLOPs, checksum **−123.851440**, and pre-timing f64 parity. Times are microseconds.

| Actual path | Workers / block | Median | Full range | GFLOP/s | Timed boundary |
|---|---:|---:|---:|---:|---|
| Scalar dense | 1 | 20.689 | 20.659–20.739 | 7.207 | Checked dense call and output initialization; no buffer allocation/I/O |
| Prepared blocked GEMM+bias | 1 / 4 | 64.518 | 64.159–78.638 | 2.311 | Validation and bias initialization included; transpose and allocation excluded |
| Prepared blocked GEMM+bias | 1 / 16 | 23.859 | 23.839–23.870 | 6.250 | Same prepared-layout boundary |
| Prepared blocked GEMM+bias | 1 / 32 | 13.380 | 13.359–13.390 | 11.144 | Same prepared-layout boundary |
| Scoped dense | 3 | 41.519 | 37.489–44.999 | 3.591 | Call-level latency includes launch/join and handle/receipt allocations; output allocation excluded |
| Checked SIMD dense, AVX2+FMA | 1 | 40.729 | 40.719–40.799 | 3.661 | Includes per-dot feature/finite checks; output allocation excluded |

Separate intervals include output allocation + call + drop with input arrays already resident: scalar **20.709 µs** (20.689–20.729), scoped **40.809 µs** (36.909–48.569), checked SIMD **40.599 µs** (40.599–40.619). Noise can reverse small differences between intervals; none is full service/network latency. The cold-array estimate is **22392 bytes**, **6.6591 FLOP/byte**; it is not measured DRAM traffic.

Only the block-32 prepared-layout variant was faster than scalar in this snapshot. Threads and checked SIMD were slower at this size. This is not a general architecture or algorithm ranking: boundaries differ, and prepared weights/launches/validation change the work counted. No fabricated speedup or predetermined winner appears in the lessons. Auto-vectorization is described as a candidate; generated assembly was not inspected for a vectorization claim. Hardware feature detection and actual explicit AVX2/FMA and AVX-512F execution were verified. Actual ARM/other-architecture SIMD and cache/perf measurements are outside this validation.

## The two numerical browser tools

Exactly two tools were added, with later chapters linking to their first homes:

1. **Chapter 25 `memory-gemm`**: three traversals of fixed X=[2,3], B=[3,3] including a block-2 odd edge. All three traces contain exactly 18 unique (row,inner,column) contributions and finish **[30,36,42,66,81,96]**. Model counts: 18 products, 36 FLOPs, 36 explicit A/B source reads. The visible address sequence distinguishes traversal. This is intentionally a source-level numerical trace, not a cache replacement/timing simulator; no real cache misses or DRAM counts are claimed.
2. **Chapter 27 `threads-lanes`**: 7 rows, 3 workers map to **[0,3),[3,6),[6,7)**. Private squared-error sums **14,77,49** total **140**, whole-batch mean **20**; mean-of-means is shown as a counterexample. 19 half-scale values with width 8 yield lane sums **[5,6,7,8,9,10,11,12]**, horizontal total **68**, tail **27**, dot **95**; widths 4 and 16 preserve total 95. Rounded grouping gives **1 versus 0**. Hypothetical AVX2-only dispatch chooses scalar; AVX2+FMA selects that modeled backend. Capability controls explicitly do not detect the actual browser CPU.

Both use native labeled keyboard-operable controls, scoped selectors, deterministic Reset, accessible text outputs, no autoplay/network, no color-only meaning and static/no-JS worked results. Formula checks are reproducible in `check_demos.cjs`. Full shared-site/browser rendering is left to root integration; no browser screenshot test is claimed here. A review caught and corrected the hypothetical option label to “Neither AVX2 nor FMA,” avoiding a false claim that x86 has no baseline SIMD.

## Experiment controls and remaining limits

`labs/s05-cpu/README.md` names each exact editable control: `common::benchmark_with` local Shape, measurement counts, `ch26::experiment` Shape/block array, chosen ch27 dense wrapper’s worker argument, `ch27::train` named configuration tuple, and ch28 check width/length lists. No undocumented dimension flags are suggested and unsupported/ambiguous flags error. Shape labels follow actual values. Previous `common::dense`, learner baseline checkpoints and all original projects remain available for comparisons.

Deliberate limits: one scalar tile level; no packing/register microkernel, persistent thread pool, ARM NEON, masked SIMD tail or fastest-backend tuner. Threaded timing is explicitly launch-inclusive, while scalar/prepared GEMM kernels allocate no buffers during arithmetic timing. Repeated checked SIMD dispatch is retained and marked as a profiling-driven future simplification. All topics are taught and actual required supported paths implemented; none is substituted by a browser simulation. Temporary authoring generator and test-output files were removed from the downloadable lab before handoff.
