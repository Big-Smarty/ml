//! Completed warmed, alternating, correctness-backed fused-kernel experiment.
use crate::host::Plan;
pub fn schedule() -> Vec<(Plan, bool)> {
    let plans = [Plan::Separate, Plan::FusedF32, Plan::Mixed];
    let mut runs = Vec::new();
    // Alternate plans within each round so one plan does not own all late samples.
    for warmup in [true, false] {
        for _ in 0..if warmup { 3 } else { 7 } {
            runs.extend(plans.map(|plan| (plan, warmup)));
        }
    }
    runs
}
// The supplied host implements feature/range detection and two separate passes.
// A private affine intermediate avoids the storage intermediate used by Separate.
pub const SHADER: &str = r#"struct DispatchParams { element_count: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dispatch_params: DispatchParams;
@compute @workgroup_size(64)
fn fused(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < dispatch_params.element_count) { output[id.x] = max(input[id.x] * 1.5 + 0.25, 0.0); }
}
"#;
