struct DispatchParams {
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
