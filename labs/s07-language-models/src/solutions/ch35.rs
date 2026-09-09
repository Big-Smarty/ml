//! Scores choose positions; values carry retrieved features. Exclude future positions before
//! maximum and normalization. Backward adds every permitted query's contribution to shared K,V.
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
    let scale = (x.d as f64).sqrt().recip();
    for i in 0..x.t {
        for j in 0..=i {
            a.probabilities[i * x.t + j] = (0..x.d)
                .map(|z| x.q[i * x.d + z] * x.k[j * x.d + z])
                .sum::<f64>()
                * scale;
        }
        let row = &mut a.probabilities[i * x.t..i * x.t + i + 1];
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum = row.iter().map(|s| (s - max).exp()).sum::<f64>();
        for p in row {
            *p = (*p - max).exp() / sum;
        }
        for j in 0..=i {
            for z in 0..x.d {
                a.output[i * x.d + z] += a.probabilities[i * x.t + j] * x.v[j * x.d + z];
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
    let scale = (x.d as f64).sqrt().recip();
    for i in 0..x.t {
        let mut dp = vec![0.; i + 1];
        for (j, p) in dp.iter_mut().enumerate() {
            for z in 0..x.d {
                *p += up[i * x.d + z] * x.v[j * x.d + z];
                g.v[j * x.d + z] += a.probabilities[i * x.t + j] * up[i * x.d + z];
            }
        }
        let dot = dp
            .iter()
            .enumerate()
            .map(|(j, p)| p * a.probabilities[i * x.t + j])
            .sum::<f64>();
        for (j, p) in dp.iter().enumerate() {
            let ds = a.probabilities[i * x.t + j] * (p - dot) * scale;
            for z in 0..x.d {
                g.q[i * x.d + z] += ds * x.k[j * x.d + z];
                g.k[j * x.d + z] += ds * x.q[i * x.d + z];
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
