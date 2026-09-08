# Chapter 32 research notes

Route: authored by Sol High after bounded Luna High research. Feature names, timestamp APIs, and f16 syntax were checked against installed wgpu/wgpu-types/naga 29.0.4 and official Vulkan references.

`Features::TIMESTAMP_QUERY` permits beginning/end pass timestamps through `ComputePassTimestampWrites`. Results must be resolved to a `QUERY_RESOLVE` buffer and then copied to `MAP_READ`; `Queue::get_timestamp_period()` converts ticks to nanoseconds. This device duration differs from the host boundary that includes encoding, submit, polling, copy, and map. The program reports repeated median/range wall measurements and timestamp sums separately.

`Features::SHADER_F16` is queried before device creation and requested only if supported. WGSL requires `enable f16;` before global declarations. The optional shader converts f32 input to f16 arithmetic and back to f32 output. A range check prevents overflow; unsupported or unsafe input uses the f32 path.

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
