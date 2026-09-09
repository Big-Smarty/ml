//! All documents stay separate. The scheduler counts updates; the report counts actual targets.
//! Weight block means by targets, so a short final block cannot outweigh a long one.
use crate::{
    capstone,
    training::{self, Trainer},
    LabResult,
};
use ::ch36::Decoder;
pub fn train(trainer: &mut Trainer, docs: &[Vec<usize>]) -> LabResult<(f32, usize)> {
    let targets = capstone::targets(trainer, docs, 2)?;
    let loss = training::advance(
        trainer,
        docs,
        trainer.model.config().context,
        2,
        crate::solutions::ch38::update,
        crate::solutions::ch38::combine,
    )?;
    Ok((loss, targets))
}
pub fn evaluate(model: &Decoder, docs: &[Vec<usize>]) -> LabResult<f32> {
    let blocks = capstone::evaluation_blocks(model, docs)?;
    let count = blocks.iter().map(|(_, n)| n).sum::<usize>();
    Ok(blocks
        .iter()
        .map(|(loss, n)| loss * (*n as f32))
        .sum::<f32>()
        / count as f32)
}
pub fn run(args: &[String]) -> LabResult {
    capstone::run_with(args, train, evaluate)
}
pub fn check() -> LabResult {
    capstone::check_with(train, evaluate)
}
