//! The full hidden gradient and both stateful updates.
//! Gradients use unchanged W2. Adam counts once per vector update, not once per parameter.
use crate::{
    data::{Dataset, CLASSES},
    mlp::{Kind, Mlp, Optimizer},
};
pub fn gradient(
    model: &Mlp,
    data: &Dataset,
    start: usize,
    end: usize,
) -> Result<(f64, Vec<f64>), String> {
    if data.in_features() != model.input || start >= end || end > data.len() {
        return Err("invalid model/data batch".into());
    }
    let mut gradient = vec![0.; model.parameters.len()];
    let mut loss = 0.;
    let (weights1, bias1, weights2, bias2) = model.parameter_offsets();
    for n in start..end {
        let image = data.image(n);
        let (hidden, logits) = model.forward(image);
        if hidden.iter().chain(&logits).any(|value| !value.is_finite()) {
            return Err("forward computation overflowed".into());
        }
        loss += Mlp::cross_entropy_from_logits(logits, data.labels[n] as usize);
        let mut logit_gradient = Mlp::probabilities(logits);
        logit_gradient[data.labels[n] as usize] -= 1.;
        let mut hidden_gradient = vec![0.; model.hidden];
        for c in 0..CLASSES {
            gradient[bias2 + c] += logit_gradient[c];
            for j in 0..model.hidden {
                gradient[weights2 + c * model.hidden + j] += logit_gradient[c] * hidden[j];
                hidden_gradient[j] +=
                    logit_gradient[c] * model.parameters[weights2 + c * model.hidden + j]
            }
        }
        for j in 0..model.hidden {
            if hidden[j] > 0. {
                gradient[bias1 + j] += hidden_gradient[j];
                for k in 0..model.input {
                    gradient[weights1 + j * model.input + k] += hidden_gradient[j] * image[k]
                }
            }
        }
    }
    let batch_len = (end - start) as f64;
    for value in &mut gradient {
        *value /= batch_len
    }
    if !loss.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
        return Err("loss or gradient overflowed".into());
    }
    Ok((loss / batch_len, gradient))
}

pub fn update(
    optimizer: &mut Optimizer,
    parameters: &mut [f64],
    gradient: &[f64],
) -> Result<(), String> {
    optimizer.validate(parameters, gradient)?;
    if parameters.len() != gradient.len()
        || parameters.len() != optimizer.m.len()
        || parameters.len() != optimizer.v.len()
    {
        return Err("optimizer state length mismatch".into());
    }
    optimizer.step_count = optimizer
        .step_count
        .checked_add(1)
        .ok_or("optimizer step overflow")?;
    for i in 0..parameters.len() {
        optimizer.m[i] = optimizer.beta1 * optimizer.m[i]
            + (if optimizer.kind == Kind::Adam {
                1. - optimizer.beta1
            } else {
                1.
            }) * gradient[i];
        if optimizer.kind == Kind::Adam {
            optimizer.v[i] = optimizer.beta2 * optimizer.v[i]
                + (1. - optimizer.beta2) * gradient[i] * gradient[i];
            let mh = optimizer.m[i] / (1. - optimizer.beta1.powf(optimizer.step_count as f64));
            let vh = optimizer.v[i] / (1. - optimizer.beta2.powf(optimizer.step_count as f64));
            parameters[i] -= optimizer.learning_rate * mh / (vh.sqrt() + optimizer.eps)
        } else {
            parameters[i] -= optimizer.learning_rate * optimizer.m[i]
        }
    }
    if parameters
        .iter()
        .chain(&optimizer.m)
        .chain(&optimizer.v)
        .any(|v| !v.is_finite())
    {
        return Err("optimizer produced nonfinite parameters or moment buffers".into());
    }
    Ok(())
}
pub fn run(args: &[String]) -> Result<(), String> {
    crate::ch11::report(args, gradient, update)
}
pub fn check() -> Result<(), String> {
    crate::ch11::verify(gradient, update)
}
