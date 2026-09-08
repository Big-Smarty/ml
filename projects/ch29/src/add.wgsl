struct DispatchParams {
    element_count: u32,
    _pad0: u32,
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
        output[i] = left[i] + right[i];
    }
}
