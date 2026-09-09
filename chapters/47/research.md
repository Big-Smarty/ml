## Active redesign route (Section 09)

The active learner route is `just lab 47` → edit `labs/s09-advanced/src/ch47.rs` → `just lab-check 47`. The lesson now contains 2 sessions and 6 ordered question-led steps. Goal: Prune deterministically, construct CSR, verify two-input parity, and distinguish byte and timing savings. The separately explained solution is `labs/s09-advanced/src/solutions/ch47.rs`; supplied validation, fixtures, and I/O are in `src/common/ch47.rs`. Original `projects/ch47` implementations remain preserved reference evidence. Earlier workflow descriptions below are historical reference notes and do not replace this active route.

Every session includes a concrete worked calculation, implementation checkpoint, graduated hints, an explained answer, and a changed-input transfer task. The browser illustration in chapter 48 executes only a fixed local arithmetic fixture; no browser control claims to run Rust. Executable goal checks and source-review limits are recorded in `guidance/redesign/section-09.md`.

# Chapter 47 research record

## Route and verification

The author commissioned the bounded `research47` subagent using GPT-5.6 Luna High. It returned primary sources for magnitude pruning, CSR storage, sparse matrix-vector kernels, and benchmark discipline. The author verified that the final project separates pruning error, representation parity, conversion, and timed kernels.

## Evidence used

Han et al. describe a train–prune–retrain pipeline that removes low-magnitude connections and retrains surviving weights: <https://proceedings.neurips.cc/paper/2015/hash/ae0eb3eed39d2bcef4622b2499a05fe6-Abstract.html>. The course isolates the middle operation with deterministic global magnitude ranking. It does not transfer the paper's compression numbers to this unrelated fixture.

Gale, Elsen, and Hooker's large-scale study shows that simple magnitude pruning is often competitive while results depend on model, schedule, and whether sparsity is imposed during training: <https://arxiv.org/abs/1902.09574>. This supports evaluating output or task quality rather than declaring small weights disposable by definition.

Intel's oneMKL documentation specifies CSR through values, column indices, and row offsets: <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/sparse-storage-formats.html>. The Rust layout uses zero-based indices, permits empty rows, and checks that the last pointer equals the stored nonzero count by construction.

Bell and Garland analyze why sparse matrix-vector performance depends on memory traffic, locality, row balance, and format: <https://www.nvidia.com/docs/IO/77944/sc09-spmv-throughput.pdf>. Rust's `black_box` documentation warns that the barrier is best effort rather than a correctness or benchmark facility: <https://doc.rust-lang.org/std/hint/fn.black_box.html>.

## Decisions and checks

The dense oracle multiplies the masked matrix, and CSR is built from exactly that mask. Their tolerance check establishes representation parity. A separate RMSE against the unpruned output quantifies how much pruning changed this one calculation. The timed loops reuse output buffers; pruning, sorting, conversion, and allocation are outside the kernel intervals. Each path is warmed once, then run 31 times with median and full range reported.

The consistency revision retains Chapter 41's `dense_matvec_reference(weights, out_features, in_features, input)` name, argument order, and row-major [out_features,in_features] operation. Chapter 47 deliberately uses f64 and returns `Result<Vec<f64>, &'static str>` so its existing shape and finite-value checks remain at the boundary; Chapter 41's f32 oracle returns an unchecked `Vec<f32>`. CSR rows are output features and columns are input features. The allocating dense/CSR methods serve correctness checks, while the timed kernels write to separate caller-owned buffers.

No speedup threshold is asserted. The default matrix and scalar Rust loops are pedagogical, and results depend on release mode and machine. The project contains no SIMD, batching, GPU sparse library, task dataset, or sparse retraining.

## Astra High review and correction — 2026-09-08

The user-directed ownership is GPT-6 Astra High, reviewing the existing Sol High draft and implementing corrections. Bounded read-only mathematical/source verification was delegated to GPT-5.6 Luna High (`verify_math`), which browsed original Switch, S4, linear-attention, AEVB, GAN, and DDIM sources. The author integrated its evidence and independently inspected all chapter lecture, metadata, reference code, starters, and Rustlings exercise/solution files.

Reviewed global magnitude ranking, CSR empty rows, sparse indexing, and the allocation-free benchmark. Added a hand-worked storage crossover and the reason pruning output RMSE need not be monotonic. Both timed in-place kernels now have an explicit [4,0,9] oracle test. The starter retains a working dense calculation instead of an unrelated DPO demonstration.
