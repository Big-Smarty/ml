# Chapter 29 research notes — redesign version 2

## Active redesign route and verified evidence

One GPT-6 Astra High owner implemented the entire section 29–32 under the active
AUTHORING.md contract. The original lesson, metadata, research notes, reference
Rust/WGSL and starter were read before redesign. Existing reference projects
remain unchanged. The supplied systems 20–39, interactive and learning-science
research settled the design; no new research delegation was needed.

The new `labs/s06-gpu` package reuses the pinned wgpu 29.0.4 API and the original
reference algorithms through one supplied host. Learner work lives in actual
`src/ch29.rs` CPU/WGSL code, with separately explained solutions. CPU baselines
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


Route: authored by Sol High after bounded Luna High research. API names were checked against locally installed `wgpu 29.0.4`, `wgpu-types 29.0.4`, and the upstream v29 source. wgpu 29 declares Rust 1.87 as its minimum, compatible with the course's stable Rust 1.96 target.

Primary findings: `Instance::enumerate_adapters` is asynchronous in v29; an `InstanceDescriptor` can restrict enumeration to Vulkan. `AdapterInfo` exposes backend and device type, so CPU adapters can be rejected and discrete hardware preferred explicitly. `request_device` takes one `DeviceDescriptor`. Buffer mapping remains callback-based; native progress is driven with `Device::poll(PollType::wait_indefinitely())`. Storage output must be copied to a `COPY_DST | MAP_READ` staging buffer before CPU access.

Verified sources:

- https://github.com/gfx-rs/wgpu/blob/v29/examples/standalone/01_hello_compute/src/main.rs
- https://github.com/gfx-rs/wgpu/blob/v29/examples/standalone/01_hello_compute/src/shader.wgsl
- https://github.com/gfx-rs/wgpu/blob/v29/wgpu/src/api/instance.rs
- https://www.w3.org/TR/WGSL/

No third-party code or prose was copied. The vectors, WGSL, examples, and diagram descriptions are course-authored.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered wgpu 29, WGSL/Vulkan requirements, and measurement scope. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.
