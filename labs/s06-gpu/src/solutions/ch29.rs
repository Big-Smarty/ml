//! Implement a complete scaled vector operation in Rust and WGSL, then compare odd lengths.
use crate::Result;
pub fn compute(left: &[f32], right: &[f32], scale: f32) -> Result<Vec<f32>> {
    crate::validate_pair(left, right)?;
    if !scale.is_finite() {
        return Err("scale must be finite".into());
    }
    // Each lane owns exactly one output; the CPU uses the same scale as the uniform.
    Ok(left.iter().zip(right).map(|(a, b)| scale * a + b).collect())
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

@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < dispatch_params.element_count) {
        output[i] = bitcast<f32>(dispatch_params.scale_bits) * left[i] + right[i];
    }
}
"#;
