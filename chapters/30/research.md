# Chapter 30 research notes

Route: authored by Sol High after bounded Luna High research. The shader and host APIs were checked against WGSL/WebGPU specifications, wgpu v29 examples, and installed wgpu 29.0.4 sources.

Key hazards: workgroup barriers must occur in uniform control flow. Edge lanes cannot return before the barrier; they write zero for invalid tile loads and continue. WGSL `select` evaluates both value operands, so it is not an out-of-bounds guard; the final kernel uses explicit `if` statements. Workgroup memory has no cross-workgroup visibility, so each stage reduces up to 512 values per workgroup and the host dispatches further stages until one value remains. The host validates checked products, u32 shader indexing, maximum storage binding size, maximum buffer size, and dispatch limits before resource creation.

Hardware evidence supplied by the lead: release-mode ignored tests passed on AMD Radeon RX 6950 XT, RADV NAVI21, Vulkan, Mesa 26.2.2-arch3.2, including a 17×19 by 19×33 multi-workgroup edge-tile matmul and a 777-value staged reduction. This establishes correctness for those fixtures, not a speed claim or exhaustive hardware coverage.

Verified sources:

- https://www.w3.org/TR/WGSL/
- https://www.w3.org/TR/webgpu/
- https://github.com/gfx-rs/wgpu/blob/v29/examples/features/src/hello_workgroups/README.md
- https://github.com/gfx-rs/wgpu/blob/v29/examples/features/src/hello_synchronization/README.md

All code, matrices, explanations, and worked arithmetic are original course material.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered wgpu 29, WGSL/Vulkan requirements, and measurement scope. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.
