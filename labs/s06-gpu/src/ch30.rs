//! Baseline direct dots and serial subtotals already work. Build cooperative GPU versions.
use crate::Result;
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    ch30::matmul_scalar(a, b, m, k, n)
}
pub fn reduce(values: &[f32]) -> Result<f32> {
    ch30::reduce_sum_scalar(values)
}
// Zero-fill edges; every lane reaches each barrier; guard only final stores.
pub const MATMUL: &str = r#"struct Shape {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> shape: Shape;

// Working direct dot products; replace the complete body with cooperative tiling.
@compute @workgroup_size(16, 16, 1)
fn matmul(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.y >= shape.m || id.x >= shape.n) { return; }
    var sum = 0.0;
    for (var inner = 0u; inner < shape.k; inner += 1u) {
        sum += a[id.y * shape.k + inner] * b[inner * shape.n + id.x];
    }
    c[id.y * shape.n + id.x] = sum;
}
"#;
pub const REDUCE: &str = r#"struct DispatchParams {
    element_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dispatch_params: DispatchParams;
// Working serial subtotal per group. Replace with shared scratch and a reduction tree.
@compute @workgroup_size(1)
fn reduce(@builtin(workgroup_id) group: vec3<u32>) {
    var sum = 0.0;
    for (var offset = 0u; offset < 512u; offset += 1u) {
        let i = group.x * 512u + offset;
        if (i < dispatch_params.element_count) { sum += input[i]; }
    }
    output[group.x] = sum;
}
"#;
