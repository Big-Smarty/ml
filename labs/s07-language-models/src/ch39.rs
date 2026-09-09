//! Learner capstone: integrate document-safe accumulated updates and a target-weighted evaluator.
//! This baseline genuinely trains the full decoder with one microbatch and reports one block.
use crate::{
    capstone,
    training::{self, Trainer},
    LabResult,
};
use ::ch36::Decoder;
pub fn train(trainer: &mut Trainer, docs: &[Vec<usize>]) -> LabResult<(f32, usize)> {
    crate::ensure(!docs.is_empty(), "need a training document")?;
    let first = &docs[..1];
    let targets = capstone::targets(trainer, first, 1)?;
    let loss = training::advance(
        trainer,
        first,
        trainer.model.config().context,
        1,
        crate::ch38::update,
        crate::ch38::combine,
    )?;
    Ok((loss, targets))
}
pub fn evaluate(model: &Decoder, docs: &[Vec<usize>]) -> LabResult<f32> {
    let blocks = capstone::evaluation_blocks(model, docs)?;
    Ok(blocks[0].0)
}
pub fn run(args: &[String]) -> LabResult {
    println!("working baseline: first training document, one microbatch, SGD; evaluation reports first block only");
    capstone::run_with(args, train, evaluate)
}
pub fn check() -> LabResult {
    capstone::check_with(train, evaluate)
}
