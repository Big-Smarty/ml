//! Learner: integrate the full pre-normalized attention/FFN block; differentiate LayerNorm.
use crate::{decoder, LabResult};
use ::ch36::Decoder;
pub fn block(model: &Decoder, layer: usize, x: &[f32], rows: usize) -> LabResult<Vec<f32>> {
    crate::ensure(
        layer < model.config().layers && x.len() == rows * model.config().width,
        "invalid block shape",
    )?;
    // Working baseline is the identity block: embeddings still predict through the output head.
    Ok(x.to_vec())
}
pub fn norm_backward(
    x: &[f32],
    gain: &[f32],
    up: &[f32],
    d: usize,
) -> LabResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    decoder::validate_norm(x, gain, up, d)?;
    // Baseline derivative for a fixed affine feature transform. Extend it to mean/variance paths.
    let dx = up
        .iter()
        .enumerate()
        .map(|(i, g)| g * gain[i % d])
        .collect();
    let mut dg = vec![0.; d];
    let mut db = vec![0.; d];
    for i in 0..x.len() {
        dg[i % d] += up[i] * x[i];
        db[i % d] += up[i];
    }
    Ok((dx, dg, db))
}
pub fn run(args: &[String]) -> LabResult {
    decoder::run_with(args, block)
}
pub fn check() -> LabResult {
    decoder::check_with(block, norm_backward)
}
