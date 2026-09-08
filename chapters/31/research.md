# Chapter 31 research notes

Route: authored by Sol High after bounded Luna High research. Command ordering and buffer APIs were verified against installed wgpu 29.0.4 and upstream v29 source.

The implementation keeps inputs, weights, hidden activations, logits, gradients, and losses in storage buffers. Three separate compute passes establish global ordering; `workgroupBarrier` would not synchronize different dispatches. All epochs are recorded into one command encoder, avoiding per-step mapping. Only loss history and final weights are copied to staging buffers.

Hardware evidence supplied by the lead: the shader parsed and the ignored parity test passed on AMD Radeon RX 6950 XT, RADV NAVI21, Vulkan, Mesa 26.2.2-arch3.2. The default 800-step run reduced mean cross-entropy from 0.730555 to 0.013002.

Numerical review replaced probability clamping with stable logit-space binary cross-entropy: `max(z,0) - z*y + log(1 + exp(-abs(z)))`. Its derivative is exactly `sigmoid(z)-y`, including extreme logits. The scalar oracle tests selected analytical gradients with central differences and tests finite loss at ±1000.

Verified sources:

- https://github.com/gfx-rs/wgpu/blob/v29/wgpu/src/api/compute_pass.rs
- https://github.com/gfx-rs/wgpu/blob/v29/wgpu/src/api/command_encoder.rs
- https://github.com/gfx-rs/wgpu/blob/v29/wgpu/src/api/queue.rs
- https://www.w3.org/TR/webgpu/

The nonlinear dataset, initialization, equations, Rust, and WGSL are original course material. No performance claim is made for the serial teaching reduction.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered wgpu 29, WGSL/Vulkan requirements, and measurement scope. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.
