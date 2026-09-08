# Integration contract for chapters 29–56

The earlier CPU projects are independent reference checkpoints. In the LLM phase, reuse the same actual decoder model rather than replacing it with unrelated toy stand-ins.

## GPU GEMM bridge (chapter 30 owner)

Expose `projects/ch30/src/lib.rs` as package `ch30`, alongside its normal teaching executable. The reusable API is:

```rust
pub struct Gpu;
impl Gpu {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>>;
    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize)
        -> Result<Vec<f32>, Box<dyn std::error::Error>>;
}
```

A is contiguous [m,k], B [k,n], output [m,n]. Validate nonzero dimensions, checked products and slice lengths. Implement real GPU dispatch/readback. The struct stores device/queue/pipeline. Choose a hardware Vulkan adapter, prefer discrete GPU, report which one. Other convenience methods are allowed for the chapter's own reduction and timing demonstrations. Pin cached wgpu 29.0.4 if compatible with stable Rust 1.96; inspect local crate source, not obsolete web snippets. Later packages may depend on this chapter through a path dependency behind feature `gpu`.

Do not assume GPU available in sandbox: approved host-level commands DO expose the RX6950XT. The lead performs real-device checks after delivery. CPU-only users must still compile the earlier course without wgpu.

## Decoder continuity (chapters 33–39 owner)

Expose a reusable decoder library at projects/ch36/src/lib.rs. Chapter 36 demonstrates its forward/backward training on a tiny configuration. Chapters38 and39 use it for actual AdamW training/checkpointing rather than copying a different model. Later ch40 inference and ch56 MoE capstone build on this implementation; coordinate its public API with the lead as soon as settled.

The implementation must train all decoder parameters, including embeddings, attention projections, normalization and feed-forward weights. Use explicit tensor-level operations/manual backward or tensor-level autodiff, not one heap node per scalar for a 15M model. Causal masking and gradient checks are mandatory. Support tiny default and roughly15M parameter configurations in ch39 with exact reported count and explicit long-run flags. Store optimizer/step/RNG/data cursor for resume where needed. Use a course-authored small text fixture plus explicit path to licensed user text. Training, validation and generation must be real.

Add optional `gpu` feature to use ch30::Gpu::matmul for large forward and backward matrix operations. The first bridge may copy between host and device per operation: name that transfer overhead honestly, keep CPU oracle, benchmark before claiming acceleration. This is a working bridge, not a claim of full GPU residency or efficient LLM training. Chapter31/32 separately teach residency/fusion. Do not make optional GPU a nonfunctional placeholder.

## Advanced continuation

Chapter40 must compare actual decoder cached and uncached logits, not an unrelated accumulation toy. Chapter56 must integrate sparse expert feed-forward computation into a contextual language model, train the router/experts, and compare to a dense baseline. A single-step scalar routing example alone is not the capstone. Prefer reusing/extending the ch36 decoder after its owner has completed work; shared changes require lead coordination and regression checks. Large/production-scale claims require measured evidence, and are not prerequisites for the tiny reference demo.

For top-1 MoE routing, normalizing a single selected probability to one destroys the task-loss gradient into the router. Use a documented differentiable selected gate (e.g. the full-softmax probability of the selected expert), with the discrete selection treated piecewise and a separately derived balancing term. Check router and expert parameter changes during the actual contextual language-model training. Do not describe a router as trained if only the experts receive gradients.

Every numerical result quoted as measured must come from the final executable, not mental estimates. Root review found stale condition numbers and inertia values in an early draft; authors must reconcile arithmetic, fixtures, and printed output before handoff.
