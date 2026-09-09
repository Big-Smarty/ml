# Chapter 27 research

Model route: Sol High author with a bounded GPT-5.6 Luna High primary-source researcher. Current stable Rust documentation verified `thread::scope` borrowing and `split_at_mut` ownership. The Demmel/Nguyen and Arteaga et al. papers support the claim that floating-point parallel reduction order affects reproducibility and describe stronger algorithms. The project chooses private gradient shards and ascending handle reduction as the smallest scheduling-independent design for a fixed partition. The two-row gradient arithmetic, scalar/parallel inference agreement, uneven 31-row shard test, and actual loss-reducing training loop were executed locally. No speedup claim is made because launch overhead and workload size must be measured on the target system.


## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 25–28; the researcher supplied checks and source findings, while Astra implemented the corrections.

Replaced the starter’s all-zero residual test with a nonzero case. Added hand-checked loss/gradient/update arithmetic and finite input/model/output/reduction checks. Overflowed proposed updates are rejected before mutation. Qualified thread-count effects as possible rather than inevitable; fixed partition/handle order remains the reproducibility contract.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.

## Consistency revision, 2026-09-08

Aligned the one-output model with Chapter 25's frozen dense layout: `inputs=[batch,in_features]`, the single `Model.weights` row `[in_features]`, scalar bias, and flattened `predictions=[batch]`. The task operation is now `predict`; `predict_batch` and `predict_batch_parallel` are its batch wrappers. Replaced the ambiguous fused `gradient_parallel` triple with scalar and parallel `loss_and_gradient* -> (loss, Gradient)` APIs. Workers return loss and parameter-gradient sums, fixed-order parent reduction combines them, and one shared boundary divides by the complete batch size. The 17-row scalar/parallel comparison and 31-row, three-shard training test exercise unequal chunks.


## Active redesign, 2026-09-09

Owner route: one GPT-6 Astra High implementation owner for the complete CPU section; no further authorship delegation. Read the old lesson, metadata, research, reference and starter code before replacing instruction. Original `projects/ch25`–`ch28` are preserved. Historical TODO/Rustlings statements above describe the earlier course, not the active lab.

Reused the already checked Rust/LLVM and original research sources recorded above; no new claim of fresh web verification is made. The systems audit, interactive design evidence and learning-science synthesis informed worked arithmetic, 30–45-minute sessions, meaningful core-algorithm work, progressive hints, and unfamiliar-input transfer. These are design inferences, not a claim that this Rust course has demonstrated a causal learning benefit.

The active learner entry is `labs/s05-cpu/src/ch27.rs`, with a complete separate solution. The section report `guidance/redesign/section-05.md` records exact topics/step mapping, actual local checks and benchmark configurations, prerequisite audit, hardware limits and manual source criteria. No browser illustration predicts hardware timing or substitutes for a Rust/hardware check.
