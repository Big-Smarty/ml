# Chapter 31 research notes — redesign version 2

## Active redesign route and verified evidence

One GPT-6 Astra High owner implemented the entire section 29–32 under the active
AUTHORING.md contract. The original lesson, metadata, research notes, reference
Rust/WGSL and starter were read before redesign. Existing reference projects
remain unchanged. The supplied systems 20–39, interactive and learning-science
research settled the design; no new research delegation was needed.

The new `labs/s06-gpu` package reuses the pinned wgpu 29.0.4 API and the original
reference algorithms through one supplied host. Learner work lives in actual
`src/ch31.rs` CPU/WGSL code, with separately explained solutions. CPU baselines
run without adapter discovery; hardware is an explicit `--gpu` request. Numeric
check results, manual structure criteria, measured host/device intervals and
limits are documented in `guidance/redesign/section-06.md`.

On 2026-09-09 the new lab's real release hardware suite passed on AMD Radeon
RX 6950XT/RADV NAVI21/Vulkan/Mesa 26.2.2-arch3.2. This is newly executed evidence,
not inherited reference output. All four hardware baselines also ran. A forced
no-driver experiment returned explicit unsupported rather than software
fallback. Metadata topic keys and ordered steps preserve the original course
scope. Guided worked examples, prediction, meaningful implementation and
transfer practice follow the supplied evidence synthesis as a design inference;
no claim is made that these widgets or passing tests demonstrate mastery.

All fixtures are course-authored. Original primary-source links below remain
applicable to the pinned API. Existing excerpts were retained and checked
against source; historical implementation/validation notes are separated below
because their default run and old authoring route describe preserved projects,
not the new CPU-first lab.

## Historical reference research


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
