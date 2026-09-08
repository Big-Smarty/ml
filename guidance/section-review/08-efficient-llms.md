# Section 08 review — Efficient and adapted LLMs

Reviewed 2026-09-08 by the assigned section reviewer, against base commit `43a16cc2c0b09a2bc4c75f79869cc2b9ab4fca5c`.

## Coverage

Read every chapter 40–46 `lesson.html`, `meta.json`, and `research.md`; every corresponding reference and starter `src/main.rs` and `Cargo.toml`; and all seven `exercises/cpu/exercises/chNN_01.rs` / `solutions/chNN_01.rs` pairs. Checked the assigned topics in `course.json` and the contracts in `guidance/CONSISTENCY.md`, `NUMERICS.md`, and `AUTHORING.md`. Applied the Rust coding and OpenAI Rust pattern review skills.

Read the relevant Chapter 36 parameter spans, inference operators, generation and loss interfaces; Chapter 39 generation call sites; and Chapter 47's continuity/dense-operation interface. No shared interface changes are needed.

| Chapter | Review emphasis |
| --- | --- |
| 40 | Per-layer K/V provenance, positions, heads, projection orientation, complete-prefix logit oracle, temperature/top-k probabilities, request scheduling and context limit |
| 41 | Signed nibble packing and odd tails, row scales, [D,V] ↔ [V,D] conversion, byte counts, quantization error versus decoder loss, allocating benchmark scope |
| 42 | Online numerator/denominator rescaling, causal coverage, contiguous GQA mapping, temporary versus quadratic storage, overflow checks, mixed numerical tolerance |
| 43 | All-parameter SFT, simultaneous LoRA factor gradients, frozen base, alpha/r scaling, temperature-one soft-target derivative identity and teacher entropy |
| 44 | BM25 arithmetic, both average-embedding gradients, exact search, training/retrieval reductions, extraction versus generation, attribution and abstention limits |
| 45 | Sample-mean bandit estimates, epsilon exploration, terminal bootstrap suppression, exact chain returns, sampled policy-gradient sign and pre-update baseline |
| 46 | Frozen-reference DPO margin, mean simultaneous pair gradients, stable log-softmax/softplus, exact expected-reward gradient and checked arithmetic |

## Material findings

None. The implementations, worked numerical examples, and explanations agree within the stated teaching scope. No lesson, implementation, starter, or exercise edits were warranted. Existing limitations are explicit, including the non-fused scheduler, scalar quantization/attention kernels, tiny synthetic adaptation data, extractive answering and heuristic abstention, one-step REINFORCE, and categorical rather than autoregressive DPO.

## Validation

For each `NN` in 40–46, ran the following on both `projects/chNN/Cargo.toml` and `projects/chNN/starter/Cargo.toml`:

```text
cargo fmt --manifest-path MANIFEST --check
cargo clippy --offline --manifest-path MANIFEST --all-targets -- -D warnings
cargo test --offline --manifest-path MANIFEST
cargo run --offline --manifest-path MANIFEST
```

All 14 formatting gates, 14 strict Clippy gates, and 14 default launches passed. Reference tests passed: ch40 4, ch41 4, ch42 4, ch43 4, ch44 4, ch45 5, ch46 4 (29 total). Each starter test suite failed only at its intentional `todo!`, as required.

Compiled each exercise and solution with `rustc --edition=2021 --test PATH -o /tmp/ml-section08-checks/BINARY`, then ran the result. All seven solutions passed; all seven exercises compiled and failed only at their intentional `todo!`.

A scoped whitespace-normalized comparison checked all 20 `data-source` excerpts against their actual files: all matched. Default demo outputs reproduce the lecture's cache count/error, quantized storage/error, tiled-attention error, adaptation losses, retrieval/abstention counts, bandit/policy outcomes, and preference/reward results. Execution logs are in `/tmp/ml-section08-checks/`.

No extended training, optional timing benchmarks, GPU checks, global formatting, global site builds, or global verifiers were run. The default CPU demos validate mechanisms and their quoted fixture values; they establish no hardware speedups or real-model quality. Source research notes were read, but this final review did not re-fetch their external sources because no disputed factual claim required a new source decision.

## Handoff

Only this review report was added. No cross-section issue remains. Review complete; chapter files frozen for integration.
