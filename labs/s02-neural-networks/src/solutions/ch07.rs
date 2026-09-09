//! Complete mean backpropagation: compute every path before mutating any weight.
use crate::xor::*;
pub fn gradient(net: &Net, data: &[Example]) -> Result<Gradient, &'static str> {
    validate_data(data)?;
    let n = data.len() as f64;
    let mut gradient = Gradient::default();
    for &(features, target) in data {
        let forward = net.forward(features);
        let output_signal = forward.probability - target;
        for j in 0..2 {
            gradient.output_weights[j] += output_signal * forward.hidden[j] / n;
            let hidden_signal = output_signal
                * net.output_weights[j]
                * forward.hidden[j]
                * (1.0 - forward.hidden[j]);
            gradient.hidden_bias[j] += hidden_signal / n;
            for (weight_gradient, feature) in gradient.hidden_weights[j].iter_mut().zip(features) {
                *weight_gradient += hidden_signal * feature / n;
            }
        }
        gradient.output_bias += output_signal / n;
    }
    Ok(gradient)
}

pub fn run(args: &[String]) -> Result<(), String> {
    crate::ch07::report(args, gradient)
}
pub fn check() -> Result<(), String> {
    crate::ch07::verify(gradient)
}
