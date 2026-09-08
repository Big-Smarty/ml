# Chapter 31 research notes

Route: authored by Sol High after bounded Luna High research. Command ordering and buffer APIs were verified against installed wgpu 29.0.4 and upstream v29 source.

The implementation keeps inputs, parameters, hidden activations, logits, gradients, and losses in storage buffers. Three separate compute passes establish global ordering; `workgroupBarrier` would not synchronize different dispatches. All full-batch steps are recorded into one command encoder, avoiding per-step mapping. Only loss history and final parameters are copied to staging buffers. Each update consumes every example once, so one step is also one epoch for this fixed project.

Hardware evidence supplied by the lead: the shader parsed and the ignored parity test passed on AMD Radeon RX 6950 XT, RADV NAVI21, Vulkan, Mesa 26.2.2-arch3.2. The default 800-step run reduced mean cross-entropy from 0.730555 to 0.013002.

Numerical review replaced probability clamping with stable binary cross-entropy from logits: `max(z,0) - z*y + log(1 + exp(-abs(z)))`. This is the same objective as Chapter 4's `binary_cross_entropy_from_logit`, with the same mean-over-examples reduction and derivative `sigmoid(z)-y`. Chapter 31 uses `f32` for the scalar oracle and WGSL path rather than Chapter 4's `f64`, so comparisons use explicit tolerances. The scalar oracle tests selected analytical gradients with central differences and tests finite loss at ±1000.

Consistency revision: `Model::forward` computes the hidden cache and logit; `Model::probability` produces the task probability; `Model::loss`, `Model::gradient`, and `Model::step` are the CPU oracles for the fused backward and update kernels; `train_with_gpu(data, steps, learning_rate)` is the device-resident trainer. The packed parameter order and binding layout are unchanged. WGSL uses local `label` only because `target` is reserved; the value comes from the `targets` buffer.

Verified sources:

- https://github.com/gfx-rs/wgpu/blob/v29/wgpu/src/api/compute_pass.rs
- https://github.com/gfx-rs/wgpu/blob/v29/wgpu/src/api/command_encoder.rs
- https://github.com/gfx-rs/wgpu/blob/v29/wgpu/src/api/queue.rs
- https://www.w3.org/TR/webgpu/

The nonlinear dataset, initialization, equations, Rust, and WGSL are original course material. No performance claim is made for the serial teaching reduction.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered wgpu 29, WGSL/Vulkan requirements, and measurement scope. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.
