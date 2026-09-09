//! Learner: implement warmup/cosine, token-weighted accumulation and clipped AdamW.
//! The working baseline uses one microbatch and fixed-rate SGD on the actual decoder.
use crate::{
    training::{self, Proposal, TrainConfig},
    LabResult,
};
use ::ch36::Gradients;
pub fn rate(c: TrainConfig, _step: u64) -> f32 {
    c.peak_learning_rate
}
pub fn combine(batches: &[Gradients]) -> LabResult<Gradients> {
    training::validate_batches(batches)?;
    Ok(batches[0].clone())
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
    Ok(Proposal {
        parameters: p
            .iter()
            .zip(g)
            .map(|(p, g)| p - rate(c, step) * g)
            .collect(),
        first: m.to_vec(),
        second: v.to_vec(),
    })
}
pub fn run(args: &[String]) -> LabResult {
    training::run_with(args, update, combine)
}
pub fn check() -> LabResult {
    training::check_with(update, combine, rate)
}
