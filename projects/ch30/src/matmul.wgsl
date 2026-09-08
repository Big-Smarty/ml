struct Shape {
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
