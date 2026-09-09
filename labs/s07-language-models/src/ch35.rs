//! Learner: replace causal uniform averaging with scaled QK softmax and its complete backward.
use crate::{
    attention::{self, Attention, Gradients, Inputs},
    LabResult,
};
pub fn forward(x: &Inputs) -> LabResult<Attention> {
    attention::validate(x)?;
    let mut a = Attention {
        output: vec![0.; x.t * x.d],
        probabilities: vec![0.; x.t * x.t],
    };
    for i in 0..x.t {
        for j in 0..=i {
            let p = 1. / (i + 1) as f64;
            a.probabilities[i * x.t + j] = p;
            for z in 0..x.d {
                a.output[i * x.d + z] += p * x.v[j * x.d + z];
            }
        }
    }
    Ok(a)
}
pub fn backward(x: &Inputs, a: &Attention, up: &[f64]) -> LabResult<Gradients> {
    attention::validate_backward(x, a, up)?;
    let mut g = Gradients {
        q: vec![0.; x.q.len()],
        k: vec![0.; x.k.len()],
        v: vec![0.; x.v.len()],
    };
    for i in 0..x.t {
        for j in 0..=i {
            for z in 0..x.d {
                g.v[j * x.d + z] += a.probabilities[i * x.t + j] * up[i * x.d + z];
            }
        }
    }
    Ok(g)
}
pub fn run(args: &[String]) -> LabResult {
    attention::run_with(args, forward, backward)
}
pub fn check() -> LabResult {
    attention::check_with(forward, backward)
}
