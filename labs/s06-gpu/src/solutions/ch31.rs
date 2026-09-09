//! Completed 2–4–1 training: all gradients use the same old parameter version.
use crate::{
    model::{Example, Gradient, Model},
    Result,
};
pub fn gradient(model: &Model, data: &[Example]) -> std::result::Result<Gradient, &'static str> {
    crate::model::validate_data(data)?;
    let mut gradient = Gradient {
        parameters: [0.0; 17],
    };
    for &(features, target) in data {
        let (hidden, _) = model.forward(features);
        let output_delta = model.probability(features) - target;
        for (unit, &activation) in hidden.iter().enumerate() {
            gradient.parameters[12 + unit] += output_delta * activation;
            // Chain rule uses the old output weight and tanh derivative.
            let hidden_delta =
                output_delta * model.parameters[12 + unit] * (1.0 - activation * activation);
            gradient.parameters[2 * unit] += hidden_delta * features[0];
            gradient.parameters[2 * unit + 1] += hidden_delta * features[1];
            gradient.parameters[8 + unit] += hidden_delta;
        }
        gradient.parameters[16] += output_delta;
    }
    for value in &mut gradient.parameters {
        *value /= data.len() as f32;
    }
    Ok(gradient)
}

pub fn train(mut model: Model, data: &[Example], steps: usize, rate: f32) -> Result<Model> {
    crate::model::validate_data(data)?;
    if steps == 0 || !rate.is_finite() || rate <= 0.0 {
        return Err("steps/rate must be positive and finite".into());
    }
    for _ in 0..steps {
        let gradient = gradient(&model, data)?;
        for (parameter, slope) in model.parameters.iter_mut().zip(gradient.parameters) {
            *parameter -= rate * slope;
        }
    }
    Ok(model)
}
pub const SHADER: &str = r#"const HIDDEN: u32 = 4u;
const PARAM_COUNT: u32 = 17u;

struct Config {
    samples: u32,
    history_index: u32,
    learning_rate_bits: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
// 0..8 hidden input weights, 8..12 hidden biases,
// 12..16 output weights, and 16 output bias.
@group(0) @binding(2) var<storage, read_write> parameters: array<f32>;
@group(0) @binding(3) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;
@group(0) @binding(5) var<storage, read_write> gradients: array<f32>;
@group(0) @binding(6) var<storage, read_write> losses: array<f32>;
@group(0) @binding(7) var<uniform> config: Config;

fn sigmoid(logit: f32) -> f32 {
    if (logit >= 0.0) { return 1.0 / (1.0 + exp(-logit)); }
    let exponential = exp(logit);
    return exponential / (1.0 + exponential);
}

@compute @workgroup_size(64)
fn forward(@builtin(global_invocation_id) id: vec3<u32>) {
    let sample = id.x;
    if (sample >= config.samples) { return; }
    let x0 = inputs[sample * 2u];
    let x1 = inputs[sample * 2u + 1u];
    var output_logit = parameters[16u];
    for (var unit = 0u; unit < HIDDEN; unit += 1u) {
        let value = tanh(parameters[unit * 2u] * x0 + parameters[unit * 2u + 1u] * x1 + parameters[8u + unit]);
        hidden[sample * HIDDEN + unit] = value;
        output_logit += parameters[12u + unit] * value;
    }
    logits[sample] = output_logit;
}

// ponytail: serial batch accumulation; reduce private per-example gradients for larger batches.
@compute @workgroup_size(1)
fn backward() {
    for (var parameter = 0u; parameter < PARAM_COUNT; parameter += 1u) {
        gradients[parameter] = 0.0;
    }
    var loss = 0.0;
    for (var sample = 0u; sample < config.samples; sample += 1u) {
        let x0 = inputs[sample * 2u];
        let x1 = inputs[sample * 2u + 1u];
        // WGSL reserves `target`; `label` is this example's binary target.
        let label = targets[sample];
        let logit = logits[sample];
        let probability = sigmoid(logit);
        loss += max(logit, 0.0) - logit * label + log(1.0 + exp(-abs(logit)));
        let output_delta = probability - label;
        for (var unit = 0u; unit < HIDDEN; unit += 1u) {
            let activation = hidden[sample * HIDDEN + unit];
            gradients[12u + unit] += output_delta * activation;
            let hidden_delta = output_delta * parameters[12u + unit] * (1.0 - activation * activation);
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
    losses[config.history_index] = loss * inverse_n;
}

@compute @workgroup_size(1)
fn update() {
    let learning_rate = bitcast<f32>(config.learning_rate_bits);
    for (var parameter = 0u; parameter < PARAM_COUNT; parameter += 1u) {
        parameters[parameter] -= learning_rate * gradients[parameter];
    }
}
"#;
