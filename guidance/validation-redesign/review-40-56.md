# Independent review: Chapters 40–56

Date: 2026-09-09
Scope: redesigned lessons, metadata, research records, learner labs, explained solutions, goal checks, browser arithmetic, and section reports for Chapters 40–56. The preserved `projects/ch40`–`projects/ch56` implementations were treated as read-only references.

## Verdict

No unresolved P1 or P2 issue remains in Chapters 40–56. Both section packages pass strict Rust checks and the scoped course audit:

| Scope | Ordered steps | Rust and browser result | Scoped audit |
|---|---:|---|---|
| Chapters 40–46 / `s08-adaptation` | 69 | format and strict Clippy pass; 43 tests pass; Chapter 40 browser arithmetic passes | `Section 08: PASS`; 0 findings |
| Chapters 47–56 / `s09-advanced` | 75 | format and strict Clippy pass; 44 library tests plus 1 baseline integration test pass; Chapter 48 browser arithmetic passes | `Section 09: PASS`; 0 findings |

The redesigned work is suitable for the stated learner: an ML beginner who already programs comfortably in Rust. Each active starter runs a meaningful baseline, each exercise names the learner-owned algorithm, and each solution is separate. The checks test changed inputs or failure modes rather than a missing scalar expression. Session estimates stay within 30–45 minutes. Rustlings and retired starter paths are absent from the active route.

The implementation boundaries are also appropriate. Learners own the central numerical or systems mechanism; fixtures, repetitive tensor plumbing, parsing, bounded I/O, and checkpoint encoding are supplied where they would otherwise crowd out the chapter goal. The largest capstone still exposes and tests the complete forward, backward, training, recovery, and serving path.

## Findings resolved during review

### P1 — Chapter 40 originally hid the learning goal in one decoder-step task

The first Section 08 draft placed roughly a complete cached decoder step behind one large learner function. That was too broad for a short session and made failures hard to localize. The final learner owns three named stages—position-aware embedding, K/V projection and append, and cached attention—plus sampling and scheduling. Supplied orchestration connects those stages to the complete decoder.

`40 --stages` now reports independent evidence for position handling, retained K/V rows, hand-computable attention, and prefix parity. The complete goal check still compares all cached logits against full-prefix decoding on unfamiliar prefixes. This resolves both task size and diagnostic quality without reducing the conceptual goal.

### P1 — Chapter 55 originally reduced interchange to a transpose

The first Section 09 draft asked for little beyond a non-square layout mapping. The final learner owns the layout adapter, coordinate-wise parity diagnostics, an independent runtime forward pass, full weight and bias gradients, SGD update, three-input comparison, and the post-update parity workflow. Actual 72-byte export and separate-process import exercise the stated artifact boundary. The optional Burn program is clearly described as a separate 2×2 reference rather than evidence that the custom artifact is ONNX or generally portable.

### P2 — Active-path and lesson accuracy defects

The following review findings were corrected before the final gates:

- Chapter 42 no longer points to a retired `projects/ch42/starter` source.
- Goal failures are classified at the individual check boundary; ordinary parser, filesystem, argument, and runtime errors are not mislabeled as `GOAL_NOT_MET`.
- Chapter 49 documents its manual source review honestly. Once learner numerics pass, the CLI returns `GOAL_REVIEW_REQUIRED:` with exit 3 because numerical equality cannot distinguish the required offset-doubling schedule from a serial prefix loop. The author-reviewed solution exits 0.
- Chapter 52 labels the two two-candidate values as the losses for logits `[1,0]` and `[2,0]`, rather than calling one a symmetric loss.
- Chapter 54 now gives the correct Group B values: TPR `1/2`, FPR `1/4`, positive prediction rate `1/3`, and accuracy `2/3`. Its prerequisite references no longer assign unrelated attribution material to Chapter 35.
- Chapter 47 retrieves row-major storage from Chapter 26, and Chapter 56 retrieves the complete decoder from Chapter 36.

## Chapter-by-chapter correctness review

| Chapter | Independent conclusion |
|---|---|
| 40 | Cache state, every-prefix oracle parity, exact cache accounting, sampling, and round-robin request progress are separated and correctly checked. The scalar scheduler makes no networking or fused-batching claim. |
| 41 | Row-wise int8, signed packed int4, wide integer accumulation, metadata bytes, output error, model loss, and timing are distinguished. The prose correctly says no explicit SIMD and does not infer speed from storage. |
| 42 | Online maximum/denominator/numerator rescaling, causal tiles, uneven tile widths, and grouped-query head mapping agree with the scalar oracle. The lesson claims bounded workspace and exact arithmetic, not FlashAttention hardware speed. |
| 43 | Response-only SFT masking, both LoRA factors, frozen-base and merge parity, and temperature-consistent distillation gradients are exercised. InstructGPT results are not attributed to SFT alone. |
| 44 | BM25, learned exact dense retrieval, source-preserving context, unsupported overlap queries, abstention coverage, and selective-risk denominators are tested separately. Retrieval, citation provenance, and entailment are not conflated. |
| 45 | Epsilon-greedy bandits, terminal-aware Q-learning, REINFORCE, and clipped PPO use explicit finite fixtures and multiple checks. Seeded outcomes are presented as mechanics evidence rather than convergence guarantees. |
| 46 | Bradley–Terry reward learning, stable DPO with frozen reference and simultaneous batch gradients, and exact verifiable-reward optimization are distinct. Pair preference is not presented as correctness, and beta is not a hard KL constraint. |
| 47 | Deterministic global pruning, tie behavior, empty-row CSR, indirect SpMV, masked-dense parity, original-output damage, payload bytes, and kernel time have the correct comparators and denominators. |
| 48 | Selected full-softmax gating retains the task gradient into all router logits. Capacity uses attempted versus admitted counts correctly; dropped examples remain in the all-attempt task denominator and balance statistics. The browser illustration matches the Rust convention. |
| 49 | Ordered transition composition and immutable previous-round Hillis–Steele scans match recurrence, including nonzero state and non-power-of-two length. Recurrent feature-kernel attention matches its direct oracle and is explicitly distinguished from softmax attention. |
| 50 | Uneven data-parallel shards aggregate sums and counts, tensor partitions reconstruct a non-square computation, both pipeline stages update through real channels, and recovery includes the state needed by this SGD fixture. CPU threads are not described as multi-GPU scaling. |
| 51 | Reparameterized beta-VAE, stable non-saturating GAN objectives, and deterministic DDIM reversal are checked with derivative and perfect-noise probes. Raw objectives are not compared across model families, and the affine Gaussian family limitation is explicit. |
| 52 | Normalization, complete similarity matrices, symmetric row/column InfoNCE, both tower updates, explicit candidates, retrieval directions, and false-negative/tie limits are correct. Same-concept perturbations are not claimed as open-vocabulary generalization. |
| 53 | Preprocessing lives in the artifact prediction path; schema/version/hash validation, actual file round trip, all-canary promotion, rollback, bounded recent-window monitoring, and a real loopback request are tested. Mean drift is presented as a signal rather than quality loss. |
| 54 | Attribution reference dependence, endpoint perturbation, membership-attack denominators, subgroup confusion rates with undefined cases, strict nonfinite parsing, and Simpson-style association reversal are correctly bounded. The audit does not claim fairness, privacy, robustness, security, or causality certification. |
| 55 | The non-square `[out,in]` to `[in,out]` mapping preserves individual coordinates through forward, gradient, bias, update, and restored inference checks. Malformed length and nonfinite artifact values are rejected. |
| 56 | The capstone is an actual pre-normalized causal decoder with sparse experts, selected-probability gates, capacity, residual drops, balancing, full backpropagation, training, uncapped causal evaluation, checkpoint identity, exact resume, generation, and bounded HTTP serving. All parameter gradients are finite-differenced away from route boundaries; one expert matches an independent dense oracle. |

## Runtime evidence

The final independent commands were:

```sh
python3 tools/verify.py --chapters 40 41 42 43 44 45 46 --rust
python3 tools/verify.py --chapters 47 48 49 50 51 52 53 54 55 56 --rust

cargo fmt --manifest-path labs/s08-adaptation/Cargo.toml --check
cargo clippy --manifest-path labs/s08-adaptation/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path labs/s08-adaptation/Cargo.toml
node labs/s08-adaptation/test-demo.cjs

cargo fmt --manifest-path labs/s09-advanced/Cargo.toml --check
cargo clippy --manifest-path labs/s09-advanced/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path labs/s09-advanced/Cargo.toml
node chapters/48/demo-check.cjs
cargo test --manifest-path labs/s09-advanced/Cargo.toml serving_monitor_and_rollback_are_real -- --ignored
```

For Section 08, all seven learner baselines exit 0, all seven unfinished learner checks exit 1 with an algorithm-specific `GOAL_NOT_MET:` diagnostic, and all seven solution checks exit 0. Chapter 40’s supplied stages fail independently before implementation and all pass in the solution.

For Section 09, all ten learner baselines exit 0, all ten unfinished learner checks exit 1 at distinct algorithmic invariants, and all ten solution checks exit 0. Chapter 55 export/import succeeded across invocations. The explicit Chapter 53 loopback test accepted one request and passed its drift assertion. Chapter 56’s author evidence includes a 160-step sparse run, separate held-out evaluation, a 200 HTTP generation response, and a byte-identical interrupted/resumed versus uninterrupted 180-step checkpoint.

Detailed author transcripts and measurements are retained in:

- `guidance/validation-redesign/section-08-author.json`
- `guidance/validation-redesign/section-09-author.json`
- `guidance/redesign/section-08.md`
- `guidance/redesign/section-09.md`

## Limits of this verdict

The browser checks verify deterministic arithmetic, control effects, and static fallback content. They do not replace the root-level integrated visual, keyboard, and responsive-layout review. The optional Chapter 55 Burn dependency path was not executed in the final Section 09 validation; the default custom-runtime parity evidence does not stand in for it. Release timings in Section 08 are measurements of the stated local scalar fixtures and machine, not portable performance results.

No controlled study directly establishes that this exact end-to-end Rust ML sequence teaches the intended population. The pedagogy is a defensible application of worked examples, subgoal labels, immediate analogous practice, performance-sensitive fading, prediction and self-explanation prompts, and inspectable experimental artifacts. The lessons correctly keep far-transfer and LLM-specific learning claims provisional.
