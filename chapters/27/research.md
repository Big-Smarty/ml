# Chapter 27 research

Model route: Sol High author with a bounded GPT-5.6 Luna High primary-source researcher. Current stable Rust documentation verified `thread::scope` borrowing and `split_at_mut` ownership. The Demmel/Nguyen and Arteaga et al. papers support the claim that floating-point parallel reduction order affects reproducibility and describe stronger algorithms. The project chooses private gradient shards and ascending handle reduction as the smallest scheduling-independent design for a fixed partition. The two-row gradient arithmetic, scalar/parallel inference agreement, uneven 31-row shard test, and actual loss-reducing training loop were executed locally. No speedup claim is made because launch overhead and workload size must be measured on the target system.


## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 25–28; the researcher supplied checks and source findings, while Astra implemented the corrections.

Replaced the starter’s all-zero residual test with a nonzero case. Added hand-checked loss/gradient/update arithmetic and finite input/model/output/reduction checks. Overflowed proposed updates are rejected before mutation. Qualified thread-count effects as possible rather than inevitable; fixed partition/handle order remains the reproducibility contract.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.
