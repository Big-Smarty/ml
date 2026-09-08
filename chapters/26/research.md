# Chapter 26 research

Model route: Sol High author with a bounded GPT-5.6 Luna High primary-source researcher. Goto and van de Geijn’s original GEMM analysis and Intel’s optimization manual support the memory-hierarchy and blocking discussion. Current Rust `Vec` and slice documentation supports the allocation-reuse and safe remainder design. The course implementation deliberately stops before packed panels and register microkernels: two scalar loop organizations and one tile level isolate the first locality decisions. Hand arithmetic, nonsquare 3×5 by 5×7 output comparisons, partial blocks, `usize::MAX` overflow-safe bounds, and release benchmark execution were checked locally. Reported performance remains measured output, never a sourced or fabricated speedup.


## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 25–28; the researcher supplied checks and source findings, while Astra implemented the corrections.

Verified all three loop orders, nonsquare offsets, output clearing, saturating tile ends, and maximum-size blocks. Added shape/precision/architecture/thread/warmup/sample/timed-boundary labels and clarified conventional loop names versus the p reduction variable. No kernel algorithm change was needed.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.
