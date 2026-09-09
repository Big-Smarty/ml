# Write GPU kernels: chapters 29–32

One supplied wgpu host carries vector dispatch, staged reduction, tiled GEMM,
resident nonlinear training, and measured fusion. Every default runs tiny CPU
arithmetic. GPU execution always requires `--gpu`; a missing hardware Vulkan
adapter is an explicit unsupported error, never a software fallback or a pass.

Run from the repository root:

```bash
cargo run --manifest-path labs/s06-gpu/Cargo.toml -- 29
cargo run --manifest-path labs/s06-gpu/Cargo.toml -- 29 --check
cargo run --manifest-path labs/s06-gpu/Cargo.toml -- 29 --solution --check
cargo run --release --manifest-path labs/s06-gpu/Cargo.toml -- 29 --solution --check --gpu
```

Replace `29` with `30`, `31`, or `32`. The site also provides `just lab NN`,
`just lab-check NN`, and `just gpu NN`. `just lab-check NN --gpu` checks the
learner shader on hardware. Measure chapter 32 using the explicit release Cargo
command. `--check` exercises actual learner functions and selected shaders;
`--solution` selects separate completed files under `src/solutions/`.

| Chapter | Working baseline | Meaningful learner change | What checks establish |
|---|---|---|---|
| 29 | CPU addition; GPU serial 64-value chunks | Implement CPU SAXPY and replace the whole WGSL entry with 64 independent, bounded invocations | CPU numeric results for N=1,64,67,129 and two scales; `--gpu` checks dispatch/readback |
| 30 | Direct CPU/GPU dot products and serial subtotals | Implement cooperative reduction, shared tiles, barriers, edge guards; trace equivalent CPU stages | Odd rectangular/staged numerical agreement. Explain the implementation to establish cooperation; parity alone also passes direct kernels |
| 31 | A trained linear readout over four fixed tanh features | Implement full CPU and WGSL hidden/input gradients and preserve update ordering | All17 independent f64 derivative probes, loss reduction, four unseen points; `--gpu` checks 1/80/800 steps and a different batch |
| 32 | CPU affine/ReLU, one separate-pass hardware sample | Implement warmup/repeated alternating schedule and a complete fused WGSL kernel | Schedule design on CPU; real output/parity and timing only with `--gpu` |

The unmodified **learner** goal checks for 29,31,32 return `GOAL_NOT_MET:` (exit1) with numerical
or schedule evidence. Their baselines work. Chapter 30's direct implementation
already passes numerical checks: the learner CLI then returns
`GOAL_REVIEW_REQUIRED:` (exit3) and the concrete source/trace rubric in chapter 30
metadata. The completed solution passes automatic checks (exit0), with its
structure reviewed separately by the author.
After any learner CPU criteria pass, a CPU-only check returns exit3 until
shader/source and hardware criteria are reviewed; it never labels CPU work as
GPU completion. Chapters29/31/32 can pass the hardware checks with `--gpu` while
the source/trace rubric remains explicit. There is no source-substring test pretending to prove an algorithm. Ordinary
`cargo test` validates supplied baselines, error boundaries and completed
solution numerics, and passes. A successful baseline or diagram interaction is
not a completed learning goal.

## Files and ownership

- `src/chNN.rs`: learner CPU functions and actual WGSL strings; these are the
  source links used by the lesson.
- `src/solutions/chNN.rs`: completed algorithms with comments explaining
  indexing, gradient paths, barriers, and scheduling.
- `src/host.rs`: the single supplied hardware device/queue, allocation/binding,
  dispatch, staging map/poll, training command order and timing implementation.
  Adapted from the preserved `projects/ch29`–`projects/ch32` references.
- `src/model.rs`: supplied independent scalar 17-parameter 2–4–1 tanh oracle.
- `src/lib.rs`: fixed fixtures, numerical evidence, independent f64 gradient
  probes, run/check orchestration and timing summaries.
- `src/support/`: supplied separate affine/ReLU and feature-gated f16 kernels.

No parser, framework, general tensor abstraction, GPU interpreter, or custom
benchmark framework is required. The only local library dependency is the
preserved chapter 30 oracle; wgpu 29.0.4, bytemuck 1.25.0, and pollster 0.4.0 match
the existing references. A package-local lockfile fixes transitive versions.

## Exact variations

- **29, launch geometry:** after completing the parallel shader, change
  `@workgroup_size(64)` in the selected `ch29.rs` shader to 32 and
  `host::VECTOR_WORKGROUP_SIZE` to 32 together. N=67 then dispatches3 groups,
  still computes67 outputs and preserves every value. Restore both. The
  unrelated performance/training group sizes stay64.
- **30, identity matrix:** in the `"30"` arm of `run` in `src/lib.rs`, replace a
  matrix-check fixture with m=17,k=19,n=19 and B's values with a row-major
  identity; generate 17×19 A values. Keep CPU oracle and GPU comparison.
  Expected C=A. Add the existing odd case back before recording completion.
- **31, batch denominator:** the `"31"` arm already checks a three-row batch at
  three steps. Add a repeated copy of every row and compare one mean-gradient
  update to the original batch. Keep the same learning rate and initialization.
- **32, workload size:** change `PERFORMANCE_ELEMENT_COUNT` in `src/lib.rs`
  from 65_537 to 257. The printed N, allocations, shader bound, dispatch, CPU
  oracle, and timing all use it. Restore it before recording course-fixture
  measurements. A different size can change overhead dominance.

## Numerical contracts

All arrays are row-major f32. GEMM takes A[m,k], B[k,n] and returns C[m,n]. Dense
network input weights use the course [out,in] layout. Scalar vs GPU checks use
absolute plus relative tolerance, reject nonfinite outputs and report a failing
index. Vector:1e−6+1e−6|expected|. GEMM:1e−5+1e−4|expected|. Staged sum:
1e−4+1e−5|expected| for the bounded fixtures. Training parameters/loss:
2e−4+1e−4|expected|. The derivative test uses an independent f64 loss and central
differences with epsilon1e−5, accepting1e−6+1e−4|numeric|.

Chapter 31's loss is mean stable binary cross-entropy from logits. Every gradient
is also averaged once over the full batch; all old weights survive through
backward. Inputs, targets, parameters, hidden, logits, gradients and loss history
remain on device between passes. Only final parameters and the loss history are
mapped. The serial backward invocation is a deliberate teaching limit; private
per-example gradients plus a reduction are the upgrade for larger batches.

Chapter 32's f32 tolerance is1e−5+1e−6|expected|. The bounded f16 fixture accepts
0.01+1e−6|expected|. SHADER_F16 is queried before requesting and compiling the
half shader. The range guard selects f32 outside |x|≤40,000; that guard prevents
half overflow but is not a general accuracy guarantee. The wider check includes
50,000 and confirms f32 selection. Precision fallback is printed and stays on
the same hardware GPU; it is not a fallback adapter.

## Hardware and timing evidence

```bash
cargo fmt --manifest-path labs/s06-gpu/Cargo.toml --check
cargo clippy --manifest-path labs/s06-gpu/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path labs/s06-gpu/Cargo.toml --all-targets
cargo test --release --manifest-path labs/s06-gpu/Cargo.toml -- --ignored --nocapture
node chapters/29/demo-check.cjs
```

The ignored hardware test must execute on a real adapter to pass. It checks all
completed kernels and actual resident training. Ordinary tests leave it ignored,
explicitly unverified; they do not fabricate hardware evidence.

The chapter 32 solution schedules3 warmups and 7 measured samples per plan,
alternating plans. Pipelines are created once. Device timestamps surround only
compute passes, with no host allocation/I/O in those intervals. The summed pass
duration excludes inter-pass gaps. Tick differences multiply by the queue's
nanoseconds-per-tick period. TIMESTAMP_QUERY absence reports device time
unsupported. The narrower wall timer starts after buffers/queries are allocated,
before bind groups/encoding, and ends after output map. A separately printed
whole-call timer also includes allocations/upload and timestamp decoding.

The experiment validates every warmup and measured output. It reports median
and full observed range, not a promised speed threshold. Hardware, driver,
features, dimensions and precision appear in output. It is a microbenchmark;
no browser arithmetic, traffic count, or tiny fixture proves application speed.
Actual validation measurements and limitations are recorded in
`guidance/redesign/section-06.md`.

## Data

Everything is course-authored and offline: short deterministic vectors,
rectangular matrices, eight XOR-like training points, four held-out sign
combinations, and a bounded repeating affine/ReLU fixture. The transfer points
are distinct from training but do not constitute a representative validation
set. There is no downloading, extended training, checkpoint format, or paid
compute in this package.
