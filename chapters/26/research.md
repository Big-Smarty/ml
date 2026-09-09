# Chapter 26 research

Model route: Sol High author with a bounded GPT-5.6 Luna High primary-source researcher. Goto and van de Geijn’s original GEMM analysis and Intel’s optimization manual support the memory-hierarchy and blocking discussion. Current Rust `Vec` and slice documentation supports the allocation-reuse and safe remainder design. The course implementation deliberately stops before packed panels and register microkernels: two scalar loop organizations and one tile level isolate the first locality decisions. Hand arithmetic, nonsquare 3×5 by 5×7 output comparisons, partial blocks, `usize::MAX` overflow-safe bounds, and release benchmark execution were checked locally. Reported performance remains measured output, never a sourced or fabricated speedup.


## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 25–28; the researcher supplied checks and source findings, while Astra implemented the corrections.

Verified all three loop orders, nonsquare offsets, output clearing, saturating tile ends, and maximum-size blocks. Added shape/precision/architecture/thread/warmup/sample/timed-boundary labels. The consistency pass later named logical indices `row`, `col`, and `inner` throughout, and explicitly mapped Chapter 25's stored dense weights `[out_features,in_features]` through a transpose to GEMM B `[k,n]`. No kernel algorithm change was needed.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.


## Active redesign, 2026-09-09

Owner route: one GPT-6 Astra High implementation owner for the complete CPU section; no further authorship delegation. Read the old lesson, metadata, research, reference and starter code before replacing instruction. Original `projects/ch25`–`ch28` are preserved. Historical TODO/Rustlings statements above describe the earlier course, not the active lab.

Reused the already checked Rust/LLVM and original research sources recorded above; no new claim of fresh web verification is made. The systems audit, interactive design evidence and learning-science synthesis informed worked arithmetic, 30–45-minute sessions, meaningful core-algorithm work, progressive hints, and unfamiliar-input transfer. These are design inferences, not a claim that this Rust course has demonstrated a causal learning benefit.

The active learner entry is `labs/s05-cpu/src/ch26.rs`, with a complete separate solution. The section report `guidance/redesign/section-05.md` records exact topics/step mapping, actual local checks and benchmark configurations, prerequisite audit, hardware limits and manual source criteria. No browser illustration predicts hardware timing or substitutes for a Rust/hardware check.
