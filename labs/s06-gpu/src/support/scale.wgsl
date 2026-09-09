struct DispatchParams { element_count: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dispatch_params: DispatchParams;
@compute @workgroup_size(64)
fn scale(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < dispatch_params.element_count) { output[id.x] = input[id.x] * 1.5 + 0.25; }
}
