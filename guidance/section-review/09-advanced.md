# Section 09 review — Advanced architectures & engineering

Reviewed 2026-09-08 against base `43a16cc2c0b09a2bc4c75f79869cc2b9ab4fca5c`. Complete and frozen. No material correctness, consistency, or understandability defect found; no lesson or implementation changes made.

## Coverage

Read every `chapters/47` through `chapters/56` lesson and metadata file, all ten reference `src/main.rs` files, all ten starter `src/main.rs` files and manifests, and every corresponding `exercises/cpu/{exercises,solutions}/chNN_01.rs`. Also read the complete Chapter 55 optional framework implementation and manifest, and Chapter 56 `src/lib.rs` and `src/verification.rs`. Checked the section's `course.json` assignments and relevant prerequisite interfaces in Chapters 22, 27, 36, 38, and 41, plus the Chapter 46 transition. Consulted the shared authoring, consistency, numerical contracts and both applicable Rust review skills. Existing historical review evidence was left intact.

| Chapter | Mathematical and teaching coverage | Reference tests |
| --- | --- | --- |
| 47 | Magnitude budget/ties, CSR coordinates/empty rows, dense parity versus pruning error, storage arithmetic, benchmark scope | 3 passed |
| 48 | Selected full-softmax gate, task and balance derivatives, attempted-frequency stop-gradient, capacity and all-example reduction | 5 passed |
| 49 | Affine composition order/associativity, scan work/depth, selective coefficients, inclusive same-kernel linear-attention oracle | 4 passed |
| 50 | Uneven-worker sum/count reduction, tensor bias/residual boundary, frozen pipeline weights/backward, cursor recovery | 3 passed |
| 51 | Beta-VAE scaling/reparameterization, alternating GAN objectives, deterministic DDIM transitions, affine capacity and sampling limits | 4 passed |
| 52 | Tower orientation/normalization, symmetric row/column CE, finite-difference training, ranking and restricted evaluation | 3 passed |
| 53 | Fitted preprocessing, artifact validation/atomic save, request framing, drift denominator, startup rollback | 1 passed; host-only test also passed separately |
| 54 | Attribution baseline, perturbation endpoints, subgroup denominators, membership access/advantage, association versus identification | 2 passed |
| 55 | Hand-calculated forward/MSE/gradient/update, transpose mapping, Burn autodiff and full weight/bias restoration | 2 passed; optional Burn test also passed |
| 56 | Full causal decoder forward/backward, both residuals and router-input gradient, capacity, auxiliary reduction, all-parameter checks, dense oracle, evaluation windows, resume identity, serving | 15 library + 1 binary passed |

All 33 `data-source` code excerpts in these ten lessons match their actual source after whitespace normalization. The numerical examples and measured default outputs agree, including Chapter 48 half-MSE `1.36810 → 0.00222`, Chapter 52 symmetric CE `1.1322 → 0.0014`, and Chapter 56 dense/sparse final train CE `1.5412`/`1.3686` and held-out CE `2.0097`/`1.8787`.

## Validation

For each `NN=47..56`, ran these exact command patterns successfully:

```text
cargo fmt --manifest-path projects/chNN/Cargo.toml --check
cargo clippy --offline --manifest-path projects/chNN/Cargo.toml --all-targets -- -D warnings
cargo test --offline --manifest-path projects/chNN/Cargo.toml
cargo run --offline --release --manifest-path projects/chNN/Cargo.toml
```

For each starter, scoped formatting and offline strict all-target Clippy passed, and `cargo run --offline --manifest-path projects/chNN/starter/Cargo.toml` succeeded. All ten starter test commands returned the intended TODO failure (Chapter 51 has two tests reaching the same TODO; Chapter 53's supplied preprocessing test passes). Compiled every exercise and solution with `rustc --edition=2021 --test SOURCE -o /tmp/ml-section09-{exercises,solutions}-NN`: all compile; exercises fail solely at their intended TODOs; all eleven solution tests pass.

Additional successful checks:

- Scoped `cargo fmt --check`, offline strict all-target Clippy, and offline tests for `projects/ch55/framework/Cargo.toml`.
- `cargo test --offline --manifest-path projects/ch53/Cargo.toml -- --ignored`: real loopback request, monitor, and rollback test passed.
- Chapter 55 `export /tmp/ml-section09-hmadgtkm/dense.bin`, then optional framework release `import` of that exact artifact: forward/loss/gradient/update/record checks and three imported probes passed.
- Chapter 56 release `train /tmp/ml-section09-hmadgtkm/moe.bin 1`, `resume` for one additional step, and `generate ... rust 2` all succeeded. Started the actual release server against this checkpoint on an ephemeral loopback port; a complete GET returned HTTP 200 and the single-request process exited successfully.

Detailed command outputs are in `/tmp/ml-section09-gates.log`, `/tmp/ml-section09-practice.log`, and `/tmp/ml-section09-integration.log`. No source edits required regression tests. No GPU, extended training, global verifier/site build, git mutation, or existing data/checkpoint modification was performed. Optional integration was available entirely offline. Timing values are machine-specific; validation establishes these bounded fixtures, not large-model quality or performance.

## Remaining cross-section issues

None identified. No shared interface or serialized format changed. Only this report was added.
