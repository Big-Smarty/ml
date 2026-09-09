//! Normalize each branch input, project QKV, retrieve and project values, then add identity.
//! The second branch expands features, applies GELU and contracts before its residual addition.
use crate::{
    decoder::{self, linear, norm, weights},
    LabResult,
};
use ::ch36::Decoder;
pub fn block(model: &Decoder, layer: usize, x: &[f32], rows: usize) -> LabResult<Vec<f32>> {
    let c = model.config();
    crate::ensure(
        layer < c.layers && x.len() == rows * c.width,
        "invalid block shape",
    )?;
    let w = |name: &str| weights(model, &format!("layer.{layer}.{name}"));
    let n1 = norm(x, w("ln1_gain")?, w("ln1_bias")?)?;
    let qkv = linear(&n1, w("qkv_weight")?, w("qkv_bias")?, c.width)?;
    let context = decoder::heads(&qkv, rows, c.width, c.heads)?;
    let attention = linear(
        &context,
        w("attention_output_weight")?,
        w("attention_output_bias")?,
        c.width,
    )?;
    let residual: Vec<_> = x.iter().zip(attention).map(|(a, b)| a + b).collect();
    let n2 = norm(&residual, w("ln2_gain")?, w("ln2_bias")?)?;
    let expanded = linear(&n2, w("ff1_weight")?, w("ff1_bias")?, c.width)?;
    let activated: Vec<_> = expanded.into_iter().map(decoder::gelu).collect();
    let ff = linear(&activated, w("ff2_weight")?, w("ff2_bias")?, c.ff_width)?;
    Ok(residual.iter().zip(ff).map(|(a, b)| a + b).collect())
}
pub fn norm_backward(
    x: &[f32],
    gain: &[f32],
    up: &[f32],
    d: usize,
) -> LabResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    decoder::validate_norm(x, gain, up, d)?;
    let mut dx = vec![0.; x.len()];
    let mut dg = vec![0.; d];
    let mut db = vec![0.; d];
    for (row, (input, gradient)) in x.chunks_exact(d).zip(up.chunks_exact(d)).enumerate() {
        let mean = input.iter().sum::<f32>() / d as f32;
        let inv = (input.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32 + 1e-5)
            .sqrt()
            .recip();
        let xhat: Vec<_> = input.iter().map(|v| (v - mean) * inv).collect();
        let q: Vec<_> = gradient.iter().zip(gain).map(|(a, b)| a * b).collect();
        let mean_q = q.iter().sum::<f32>() / d as f32;
        let mean_qx = q.iter().zip(&xhat).map(|(a, b)| a * b).sum::<f32>() / d as f32;
        for j in 0..d {
            dx[row * d + j] = inv * (q[j] - mean_q - xhat[j] * mean_qx);
            dg[j] += gradient[j] * xhat[j];
            db[j] += gradient[j];
        }
    }
    Ok((dx, dg, db))
}
pub fn run(args: &[String]) -> LabResult {
    decoder::run_with(args, block)
}
pub fn check() -> LabResult {
    decoder::check_with(block, norm_backward)
}
