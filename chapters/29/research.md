# Chapter 29 research notes

Route: authored by Sol High after bounded Luna High research. API names were checked against locally installed `wgpu 29.0.4`, `wgpu-types 29.0.4`, and the upstream v29 source. wgpu 29 declares Rust 1.87 as its minimum, compatible with the course's stable Rust 1.96 target.

Primary findings: `Instance::enumerate_adapters` is asynchronous in v29; an `InstanceDescriptor` can restrict enumeration to Vulkan. `AdapterInfo` exposes backend and device type, so CPU adapters can be rejected and discrete hardware preferred explicitly. `request_device` takes one `DeviceDescriptor`. Buffer mapping remains callback-based; native progress is driven with `Device::poll(PollType::wait_indefinitely())`. Storage output must be copied to a `COPY_DST | MAP_READ` staging buffer before CPU access.

Verified sources:

- https://github.com/gfx-rs/wgpu/blob/v29/examples/standalone/01_hello_compute/src/main.rs
- https://github.com/gfx-rs/wgpu/blob/v29/examples/standalone/01_hello_compute/src/shader.wgsl
- https://github.com/gfx-rs/wgpu/blob/v29/wgpu/src/api/instance.rs
- https://www.w3.org/TR/WGSL/

No third-party code or prose was copied. The vectors, WGSL, examples, and diagram descriptions are course-authored.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered wgpu 29, WGSL/Vulkan requirements, and measurement scope. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.
