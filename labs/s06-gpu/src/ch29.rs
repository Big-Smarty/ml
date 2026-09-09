//! Implement a complete scaled vector operation in Rust and WGSL, then compare odd lengths.
use crate::Result;
pub fn compute(left: &[f32], right: &[f32], scale: f32) -> Result<Vec<f32>> {
    crate::validate_pair(left, right)?;
    if !scale.is_finite() {
        return Err("scale must be finite".into());
    }
    // Working baseline: addition. Replace the entire operation and shader with SAXPY.
    let _ = scale;
    Ok(left.iter().zip(right).map(|(a, b)| a + b).collect())
}

pub const SHADER: &str = r#"struct DispatchParams {
    element_count: u32,
    scale_bits: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dispatch_params: DispatchParams;

// Working baseline: one invocation serially adds a 64-element chunk.
// Replace this complete kernel with 64 independent invocations and scaled addition.
@compute @workgroup_size(1)
fn add(@builtin(workgroup_id) group: vec3<u32>) {
    for (var offset = 0u; offset < 64u; offset += 1u) {
        let i = group.x * 64u + offset;
        if (i < dispatch_params.element_count) { output[i] = left[i] + right[i]; }
    }
}
"#;
