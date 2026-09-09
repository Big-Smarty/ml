//! Implement a complete comparison: warmups, alternating repeated plans, and fused WGSL.
use crate::host::Plan;
pub fn schedule() -> Vec<(Plan, bool)> {
    // One completed baseline sample is useful, but cannot support a fusion comparison.
    vec![(Plan::Separate, false)]
}
// The supplied host implements feature/range detection and two separate passes.
// This identity baseline is not used by Separate; implement the fused operation before selecting it.
pub const SHADER: &str = r#"struct DispatchParams { element_count: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dispatch_params: DispatchParams;
@compute @workgroup_size(64)
fn fused(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < dispatch_params.element_count) { output[id.x] = input[id.x]; }
}
"#;
