# Chapter 32 research notes — redesign version 2

## Active redesign route and verified evidence

One GPT-6 Astra High owner implemented the entire section 29–32 under the active
AUTHORING.md contract. The original lesson, metadata, research notes, reference
Rust/WGSL and starter were read before redesign. Existing reference projects
remain unchanged. The supplied systems 20–39, interactive and learning-science
research settled the design; no new research delegation was needed.

The new `labs/s06-gpu` package reuses the pinned wgpu 29.0.4 API and the original
reference algorithms through one supplied host. Learner work lives in actual
`src/ch32.rs` CPU/WGSL code, with separately explained solutions. CPU baselines
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


Route: authored by Sol High after bounded Luna High research. Feature names, timestamp APIs, and f16 syntax were checked against installed wgpu/wgpu-types/naga 29.0.4 and official Vulkan references.

`Features::TIMESTAMP_QUERY` permits beginning/end pass timestamps through `ComputePassTimestampWrites`. Results must be resolved to a `QUERY_RESOLVE` buffer and then copied to `MAP_READ`; `Queue::get_timestamp_period()` converts ticks to nanoseconds. This device duration differs from the CPU wall-clock boundary that includes encoding, submit, polling, output copy, and output map. The CPU wall clock ends before the host maps and decodes the timestamp staging buffer. The program reports repeated median/range CPU wall measurements and device timestamp sums separately.

`Features::SHADER_F16` is queried before device creation and requested only if supported. WGSL requires `enable f16;` before global declarations. The optional shader converts f32 input to f16 arithmetic and back to f32 output. A range check prevents overflow; unsupported or unsafe input uses the f32 path.

The executable pairs `affine_relu_scalar(input)` with `Gpu::affine_relu_with_gpu(input, plan)`. It carries forward Chapter 29's `dispatch_count(element_count, workgroup_size)` and names the uniform `dispatch_params`; its `element_count` field is dispatch metadata rather than learned state. The starter and CPU exercise use `storage_value_access_count(element_count, fused)` only for the 4N-versus-2N operation-graph model, avoiding a claim that these counts are measured bus transfers.

The Vulkan extension distinction is concrete: `VK_KHR_shader_float16_int8` supplies 16-bit arithmetic, `VK_KHR_16bit_storage` supplies storage/interface capability, and `VK_KHR_cooperative_matrix` supplies subgroup-cooperative matrix operations. wgpu abstracts normal half arithmetic through SHADER_F16.

Hardware evidence supplied by the lead: on AMD Radeon RX 6950 XT with RADV NAVI21, Vulkan, Mesa 26.2.2-arch3.2, timestamp queries and shader f16 were supported; separate, fused f32, and mixed outputs passed parity. Repeated measurements cited in the lesson come from that final executable. The lower-level Vulkan extensions were researched from specifications rather than called directly.

Verified sources:

- https://github.com/gfx-rs/wgpu/blob/v29/examples/features/src/timestamp_queries/mod.rs
- https://github.com/gfx-rs/wgpu/blob/v29/wgpu-types/src/features.rs
- https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_float16_int8.html
- https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_16bit_storage.html
- https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html

No benchmark result is quoted before the final executable is run on its named hardware.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered wgpu 29, WGSL/Vulkan requirements, and measurement scope. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.
