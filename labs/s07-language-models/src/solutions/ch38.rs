//! Each microbatch mean is weighted by its targets. Clip the combined vector once, then
//! update moments. Decay uses old parameters, outside the gradient and both moments.
use crate::{
    training::{self, Proposal, TrainConfig},
    LabResult,
};
use ::ch36::Gradients;
pub fn rate(c: TrainConfig, step: u64) -> f32 {
    if step >= c.total_steps {
        return c.min_learning_rate;
    }
    if step < c.warmup_steps {
        return c.peak_learning_rate * (step + 1) as f32 / c.warmup_steps as f32;
    }
    let progress = (step - c.warmup_steps) as f32 / (c.total_steps - c.warmup_steps).max(1) as f32;
    c.min_learning_rate
        + 0.5
            * (c.peak_learning_rate - c.min_learning_rate)
            * (1. + (std::f32::consts::PI * progress).cos())
}
pub fn combine(batches: &[Gradients]) -> LabResult<Gradients> {
    training::validate_batches(batches)?;
    let tokens = batches.iter().try_fold(0usize, |n, b| {
        n.checked_add(b.tokens).ok_or("token count overflow")
    })?;
    let mut all = Gradients {
        loss: 0.,
        tokens,
        values: vec![0.; batches[0].values.len()],
    };
    for batch in batches {
        let weight = batch.tokens as f32 / tokens as f32;
        all.loss += weight * batch.loss;
        for (a, g) in all.values.iter_mut().zip(&batch.values) {
            *a += weight * g;
        }
    }
    Ok(all)
}
pub fn update(
    p: &[f32],
    m: &[f32],
    v: &[f32],
    g: &[f32],
    c: TrainConfig,
    step: u64,
) -> LabResult<Proposal> {
    training::validate_update(p, m, v, g, c, step)?;
    let norm = g.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
    let scale = (f64::from(c.clip_norm) / norm).min(1.) as f32;
    let next = (step + 1) as f32;
    let b1 = 1. - c.beta1.powf(next);
    let b2 = 1. - c.beta2.powf(next);
    let mut proposal = Proposal {
        parameters: Vec::with_capacity(p.len()),
        first: Vec::with_capacity(p.len()),
        second: Vec::with_capacity(p.len()),
    };
    for i in 0..p.len() {
        let gradient = g[i] * scale;
        let first = c.beta1 * m[i] + (1. - c.beta1) * gradient;
        let second = c.beta2 * v[i] + (1. - c.beta2) * gradient * gradient;
        let adaptive = (first / b1) / ((second / b2).sqrt() + c.epsilon);
        proposal
            .parameters
            .push(p[i] - rate(c, step) * (adaptive + c.weight_decay * p[i]));
        proposal.first.push(first);
        proposal.second.push(second);
    }
    Ok(proposal)
}
pub fn run(args: &[String]) -> LabResult {
    training::run_with(args, update, combine)
}
pub fn check() -> LabResult {
    training::check_with(update, combine, rate)
}
