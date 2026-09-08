# Chapter 25 research

Model route: Sol High author with a bounded GPT-5.6 Luna High primary-source researcher. Sources were checked against current official Rust documentation, Linux kernel documentation, and the original Berkeley Roofline technical report on 2026-09-08. The Rust pages support `f32`, `mul_add`, `black_box`, and `Instant` behavior. The kernel documentation supports `perf` as a hardware/software event interface. The Roofline report supports arithmetic intensity only when a memory level is stated. Numerical examples and the printed operation/byte formulas were checked against the chapter’s Rust implementation; measured timings remain machine-dependent and are not copied into the lesson.


## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 25–28; the researcher supplied checks and source findings, while Astra implemented the corrections.

Added an independent hand-calculated dense result, rejected non-finite oracle comparisons, fixed the starter weight shape, and printed architecture/OS/thread count. Replaced the misleading warmed-DRAM lower-bound reading with a qualified cold-array read-once/write-once traffic estimate.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.
