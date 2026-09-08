const HIDDEN: u32 = 4u;
const PARAM_COUNT: u32 = 17u;

struct Config {
    samples: u32,
    epoch: u32,
    rate_bits: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;
@group(0) @binding(5) var<storage, read_write> gradients: array<f32>;
@group(0) @binding(6) var<storage, read_write> losses: array<f32>;
@group(0) @binding(7) var<uniform> config: Config;

fn sigmoid(value: f32) -> f32 {
    return 1.0 / (1.0 + exp(-value));
}

@compute @workgroup_size(64)
fn forward(@builtin(global_invocation_id) id: vec3<u32>) {
    let sample = id.x;
    if (sample >= config.samples) { return; }
    let x0 = inputs[sample * 2u];
    let x1 = inputs[sample * 2u + 1u];
    var output_logit = weights[16u];
    for (var unit = 0u; unit < HIDDEN; unit += 1u) {
        let value = tanh(weights[unit * 2u] * x0 + weights[unit * 2u + 1u] * x1 + weights[8u + unit]);
        hidden[sample * HIDDEN + unit] = value;
        output_logit += weights[12u + unit] * value;
    }
    logits[sample] = output_logit;
}

@compute @workgroup_size(1)
fn backward() {
    for (var parameter = 0u; parameter < PARAM_COUNT; parameter += 1u) {
        gradients[parameter] = 0.0;
    }
    var loss = 0.0;
    for (var sample = 0u; sample < config.samples; sample += 1u) {
        let x0 = inputs[sample * 2u];
        let x1 = inputs[sample * 2u + 1u];
        let label = targets[sample];
        let logit = logits[sample];
        let prediction = sigmoid(logit);
        loss += max(logit, 0.0) - logit * label + log(1.0 + exp(-abs(logit)));
        let output_delta = prediction - label;
        for (var unit = 0u; unit < HIDDEN; unit += 1u) {
            let activation = hidden[sample * HIDDEN + unit];
            gradients[12u + unit] += output_delta * activation;
            let hidden_delta = output_delta * weights[12u + unit] * (1.0 - activation * activation);
            gradients[unit * 2u] += hidden_delta * x0;
            gradients[unit * 2u + 1u] += hidden_delta * x1;
            gradients[8u + unit] += hidden_delta;
        }
        gradients[16u] += output_delta;
    }
    let inverse_n = 1.0 / f32(config.samples);
    for (var parameter = 0u; parameter < PARAM_COUNT; parameter += 1u) {
        gradients[parameter] *= inverse_n;
    }
    losses[config.epoch] = loss * inverse_n;
}

@compute @workgroup_size(1)
fn update() {
    let rate = bitcast<f32>(config.rate_bits);
    for (var parameter = 0u; parameter < PARAM_COUNT; parameter += 1u) {
        weights[parameter] -= rate * gradients[parameter];
    }
}
