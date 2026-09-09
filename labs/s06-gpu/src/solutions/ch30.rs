//! Completed CPU stage traces and cooperative WGSL reduction/GEMM.
use crate::Result;
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    // Supplied oracle validates lengths, checked products and finite values first.
    let mut c = ch30::matmul_scalar(a, b, m, k, n)?;
    c.fill(0.0);
    // K blocks preserve the row-major addresses, including the partial final block.
    for base in (0..k).step_by(16) {
        for row in 0..m {
            for col in 0..n {
                for inner in base..(base + 16).min(k) {
                    c[row * n + col] += a[row * k + inner] * b[inner * n + col];
                }
            }
        }
    }
    Ok(c)
}
pub fn reduce(values: &[f32]) -> Result<f32> {
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return Err("reduction requires finite, nonempty input".into());
    }
    // This traces the same stage sizes as the GPU; within a chunk the CPU is serial.
    let mut current = values.to_vec();
    while current.len() > 1 {
        current = current
            .chunks(512)
            .map(|chunk| chunk.iter().sum())
            .collect();
    }
    Ok(current[0])
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

var<workgroup> a_tile: array<f32, 256>;
var<workgroup> b_tile: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn matmul(
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let row = group.y * 16u + local.y;
    let col = group.x * 16u + local.x;
    let lane = local.y * 16u + local.x;
    let tile_count = (shape.k + 15u) / 16u;
    var sum = 0.0;

    for (var tile = 0u; tile < tile_count; tile += 1u) {
        let a_col = tile * 16u + local.x;
        let b_row = tile * 16u + local.y;
        a_tile[lane] = 0.0;
        b_tile[lane] = 0.0;
        if (row < shape.m && a_col < shape.k) {
            a_tile[lane] = a[row * shape.k + a_col];
        }
        if (b_row < shape.k && col < shape.n) {
            b_tile[lane] = b[b_row * shape.n + col];
        }
        workgroupBarrier();

        for (var inner = 0u; inner < 16u; inner += 1u) {
            sum += a_tile[local.y * 16u + inner] * b_tile[inner * 16u + local.x];
        }
        workgroupBarrier();
    }

    if (row < shape.m && col < shape.n) {
        c[row * shape.n + col] = sum;
    }
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
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce(
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let lane = local.x;
    let first = group.x * 512u + lane;
    let second = first + 256u;
    scratch[lane] = 0.0;
    if (first < dispatch_params.element_count) { scratch[lane] = input[first]; }
    if (second < dispatch_params.element_count) { scratch[lane] += input[second]; }
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride /= 2u;
    }
    if (lane == 0u) { output[group.x] = scratch[0]; }
}
"#;
