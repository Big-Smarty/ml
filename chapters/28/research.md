# Chapter 28 research

Model route: Sol High author with a bounded GPT-5.6 Luna High primary-source researcher. The Rust Reference verified that calling a target-feature function without support is undefined behavior. Current standard-library documentation verified separate AVX2, FMA, and AVX-512F runtime checks and the stable AVX2 FMA intrinsic. Rust 1.89 release notes and a local stable Rust 1.96 compile verified that the AVX-512F intrinsics used here are stable; the outdated blanket claim that AVX-512 requires nightly is therefore excluded. LLVM documentation supports the qualified auto-vectorization discussion. The reference directly tests both AVX2/FMA and AVX-512F on capable hardware across empty, short, and odd lengths, while all generic paths retain scalar fallback behavior.


## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 25–28; the researcher supplied checks and source findings, while Astra implemented the corrections.

Verified guarded AVX2/FMA and AVX-512 plus scalar tails. Added the Rust 1.89 minimum, store safety explanations, finite oracle comparisons, a glossary link, and dispatch-inclusive timing labels. Executed both supported SIMD paths on the host; no claim about unsupported hardware or automatic vectorization was added.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.

## Consistency revision, 2026-09-08

The reference now names its checked portable boundary `dot_dispatch`, keeps `dot_scalar` as the prevalidated `f32` primitive and fallback, and names the independent higher-precision oracle `dot_f64_reference`. The starter uses the same two-layer boundary. The explicit `dot_avx512_checked` path remains separate so the default dispatch still selects only AVX2/FMA or scalar. No algorithm, benchmark scope, fixture, CLI flag, or intrinsic was changed.


## Active redesign, 2026-09-09

Owner route: one GPT-6 Astra High implementation owner for the complete CPU section; no further authorship delegation. Read the old lesson, metadata, research, reference and starter code before replacing instruction. Original `projects/ch25`–`ch28` are preserved. Historical TODO/Rustlings statements above describe the earlier course, not the active lab.

Reused the already checked Rust/LLVM and original research sources recorded above; no new claim of fresh web verification is made. The systems audit, interactive design evidence and learning-science synthesis informed worked arithmetic, 30–45-minute sessions, meaningful core-algorithm work, progressive hints, and unfamiliar-input transfer. These are design inferences, not a claim that this Rust course has demonstrated a causal learning benefit.

The active learner entry is `labs/s05-cpu/src/ch28.rs`, with a complete separate solution. The section report `guidance/redesign/section-05.md` records exact topics/step mapping, actual local checks and benchmark configurations, prerequisite audit, hardware limits and manual source criteria. No browser illustration predicts hardware timing or substitutes for a Rust/hardware check.
