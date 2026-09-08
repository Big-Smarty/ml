# Section 06 review — Write GPU kernels

Status: complete and frozen. Reviewed chapters 29–32 against base `43a16cc2c0b09a2bc4c75f79869cc2b9ab4fca5c` on 2026-09-08. No material correctness, consistency, or understandability issue found; no lesson or implementation edits were needed.

## Coverage

Read all four `chapters/NN/lesson.html`, `meta.json`, and `research.md` files; every Rust and WGSL source under `projects/chNN/src`; all four reference and starter manifests, starter implementations, and CPU exercise/solution pairs. Checked their Rustlings mappings and assigned topics in `course.json`. Applied `CONSISTENCY.md`, `NUMERICS.md`, `AUTHORING.md`, and the Rust review skills.

- **29:** Hardware-adapter selection, matching host/shader bindings and 16-byte metadata, finite/equal-length inputs, ceiling dispatch, guarded tail lanes, copy/map/poll synchronization, scalar comparison, five-element and SAXPY arithmetic.
- **30:** Rectangular row-major shapes and addresses, checked products and device limits, both tile barriers, zero-filled edges, 512-input reduction stages, the single-value reduction path, and CPU/GPU sum semantics. Verified the worked matrix product and reduction arithmetic.
- **31:** Packed 17-parameter layout, tanh/sigmoid forward values, stable logits BCE, mean-over-examples gradients, old-parameter use throughout backward, ordered updates, and 801 loss evaluations for 800 updates. CPU gradient checks and GPU parity cover the intended small fixed network; the serial backward limitation is explicit.
- **32:** Affine/ReLU values, 4N versus 2N storage-access model, asynchronous readback, exact CPU and device timing boundaries, feature-gated timestamps/f16, and overflow-range fallback. The lesson explicitly distinguishes the range guard from an accuracy guarantee and f16 arithmetic from f32 storage.

Checked prerequisite interfaces in chapters 26 and 28, the return to CPU in chapter 33, and downstream `ch30::Gpu` use in chapters 36, 38, and 39. No interface changes or cross-section issues remain. No duplicate glossary ownership involving these chapters was found.

## Validation

For each `NN` in `29 30 31 32`, ran:

```sh
cargo fmt --manifest-path projects/chNN/Cargo.toml --check
cargo clippy --offline --manifest-path projects/chNN/Cargo.toml --all-targets -- -D warnings
cargo test --offline --manifest-path projects/chNN/Cargo.toml
cargo fmt --manifest-path projects/chNN/starter/Cargo.toml --check
cargo clippy --offline --manifest-path projects/chNN/starter/Cargo.toml --all-targets -- -D warnings
cargo test --offline --manifest-path projects/chNN/starter/Cargo.toml
cargo run --offline --manifest-path projects/chNN/starter/Cargo.toml
```

All formatting and strict Clippy checks passed. Reference CPU tests passed: 29=2, 30=2, 31=4, 32=2; each retained one opt-in GPU test. All starter launches succeeded. Each starter test suite failed only at its intended guided TODO; chapter 29's additional scalar test passed.

Both exercise and solution files passed `rustfmt --check --edition 2021` and compiled with `rustc --edition 2021 --test`. Running each generated solution test binary passed its one test; each exercise binary failed its one test at the intended TODO. No incomplete learner code was changed.

A scoped HTML-parser check decoded and compared all 21 `data-source` excerpts with their named source files after whitespace normalization: 29=5, 30=6, 31=4, 32=6, all matching.

The root reviewer ran these four hardware suites sequentially:

```sh
cargo test --offline --release --manifest-path projects/chNN/Cargo.toml -- --ignored --nocapture
```

All passed on AMD Radeon RX 6950 XT (RADV NAVI21), Vulkan, Mesa 26.2.2-arch3.2; chapter 32 reported both timestamp-query and shader-f16 support. Inspected the recorded output in [gpu-checks.json](gpu-checks.json). Coverage includes a 67-element vector, odd/multiple-workgroup matrix tiles, staged reduction, resident training parity at two dataset sizes, and separate/fused/mixed output parity.

No extended training or new timing benchmark was run. Existing lecture benchmark numbers and prior 800-step evidence were preserved rather than represented as fresh measurements. Hardware results establish these fixtures on this device, not all adapters or production throughput.
