//! Supplied storage, matrix operations and reference training. Chapter 36 owns block integration.
use crate::{ensure, LabResult};
use ::ch36::{Config, Decoder};
pub type Block = fn(&Decoder, usize, &[f32], usize) -> LabResult<Vec<f32>>;
pub type NormBackward =
    fn(&[f32], &[f32], &[f32], usize) -> LabResult<(Vec<f32>, Vec<f32>, Vec<f32>)>;
pub fn config() -> Config {
    Config {
        vocab_size: 8,
        context: 4,
        width: 4,
        heads: 2,
        layers: 2,
        ff_width: 7,
    }
}
pub fn weights<'a>(model: &'a Decoder, name: &str) -> LabResult<&'a [f32]> {
    let span = model
        .parameter_spans()
        .into_iter()
        .find(|s| s.name == name)
        .ok_or_else(|| format!("unknown parameter span {name}"))?;
    Ok(&model.parameters()[span.start..span.end])
}
pub fn linear(x: &[f32], w: &[f32], b: &[f32], input: usize) -> LabResult<Vec<f32>> {
    ensure(
        input > 0 && !b.is_empty() && x.len().is_multiple_of(input) && w.len() == input * b.len(),
        "projection shape mismatch",
    )?;
    let mut out = vec![0.; x.len() / input * b.len()];
    for (row, result) in x.chunks_exact(input).zip(out.chunks_exact_mut(b.len())) {
        for (j, y) in result.iter_mut().enumerate() {
            *y = b[j]
                + row
                    .iter()
                    .enumerate()
                    .map(|(k, a)| a * w[k * b.len() + j])
                    .sum::<f32>();
        }
    }
    Ok(out)
}
pub fn norm(x: &[f32], gain: &[f32], bias: &[f32]) -> LabResult<Vec<f32>> {
    let d = gain.len();
    ensure(
        d > 0 && bias.len() == d && x.len().is_multiple_of(d),
        "normalization shape mismatch",
    )?;
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(d) {
        let mean = row.iter().sum::<f32>() / d as f32;
        let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
        out.extend(
            row.iter()
                .enumerate()
                .map(|(i, v)| (v - mean) / (variance + 1e-5).sqrt() * gain[i] + bias[i]),
        );
    }
    Ok(out)
}
pub fn heads(qkv: &[f32], t: usize, d: usize, h: usize) -> LabResult<Vec<f32>> {
    ensure(
        h > 0 && d.is_multiple_of(h) && qkv.len() == t * 3 * d,
        "QKV head shape mismatch",
    )?;
    let dh = d / h;
    let mut out = vec![0.; t * d];
    for head in 0..h {
        let part = |p: usize| -> Vec<f64> {
            (0..t)
                .flat_map(|i| {
                    (0..dh).map(move |z| f64::from(qkv[i * 3 * d + p * d + head * dh + z]))
                })
                .collect()
        };
        let x = crate::attention::Inputs {
            q: part(0),
            k: part(1),
            v: part(2),
            t,
            d: dh,
        };
        let a = crate::solutions::ch35::forward(&x)?;
        for i in 0..t {
            for z in 0..dh {
                out[i * d + head * dh + z] = a.output[i * dh + z] as f32;
            }
        }
    }
    Ok(out)
}
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1. + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}
pub fn forward_with(model: &Decoder, tokens: &[usize], block: Block) -> LabResult<Vec<f32>> {
    let c = model.config();
    ensure(
        !tokens.is_empty() && tokens.len() <= c.context && tokens.iter().all(|&i| i < c.vocab_size),
        "invalid token sequence",
    )?;
    let e = weights(model, "token_embedding")?;
    let p = weights(model, "position_embedding")?;
    let mut x = Vec::new();
    for (i, &token) in tokens.iter().enumerate() {
        for z in 0..c.width {
            x.push(e[token * c.width + z] + p[i * c.width + z]);
        }
    }
    for layer in 0..c.layers {
        x = block(model, layer, &x, tokens.len())?;
        ensure(
            x.len() == tokens.len() * c.width,
            "block must preserve [T,D]",
        )?;
    }
    let n = norm(
        &x,
        weights(model, "final_norm_gain")?,
        weights(model, "final_norm_bias")?,
    )?;
    let logits = linear(
        &n,
        weights(model, "output_weight")?,
        weights(model, "output_bias")?,
        c.width,
    )?;
    ensure(
        logits.iter().all(|v| v.is_finite()),
        "nonfinite decoder logits",
    )?;
    Ok(logits)
}
pub fn validate_norm(x: &[f32], gain: &[f32], up: &[f32], d: usize) -> LabResult {
    ensure(
        d > 0
            && !x.is_empty()
            && x.len().is_multiple_of(d)
            && gain.len() == d
            && up.len() == x.len()
            && x.iter().chain(gain).chain(up).all(|v| v.is_finite()),
        "invalid normalization backward input",
    )
}
pub fn run_with(args: &[String], block: Block) -> LabResult {
    crate::no_args(args)?;
    let mut model = Decoder::new(config(), 36).map_err(|e| e.to_string())?;
    let input = [1, 2, 3, 1];
    let target = [2, 3, 1, 2];
    println!(
        "[T,D]=[4,4], heads=2, head width=2, FF=[4,7], layers=2, parameters={}",
        model.parameter_count()
    );
    println!(
        "learner first logits={:.5?}",
        &forward_with(&model, &input, block)?[..8]
    );
    let before = model.loss(&input, &target).map_err(|e| e.to_string())?;
    for _ in 0..40 {
        let g = model
            .loss_and_gradient(&input, &target)
            .map_err(|e| e.to_string())?;
        model.apply_sgd(&g, 0.08).map_err(|e| e.to_string())?;
    }
    println!(
        "supplied complete decoder trains all families: loss={before:.5}->{:.5}",
        model.loss(&input, &target).map_err(|e| e.to_string())?
    );
    Ok(())
}
pub fn check_with(block: Block, backward: NormBackward) -> LabResult {
    let mut model = Decoder::new(config(), 36).map_err(|e| e.to_string())?;
    // Make attention/FF paths visible instead of accepting agreement between near-zero branches.
    for (i, p) in model.parameters_mut().iter_mut().enumerate() {
        *p += (i % 17) as f32 * 0.007 - 0.04;
    }
    let input = [1, 5, 3, 7];
    let actual = forward_with(&model, &input, block)?;
    let expected = model.forward(&input).map_err(|e| e.to_string())?;
    let error = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    ensure(error<2e-5,format!("goal: complete two-layer decoder forward differs by {error:.6}; implement both pre-norm branches and residual paths"))?;
    let changed = forward_with(&model, &[1, 2, 6, 4], block)?;
    ensure(
        actual[..8] == changed[..8],
        "goal: prefix logits depend on future tokens",
    )?;
    let x = [0.7, -1.2, 0.4, 2., -0.3, 0.8];
    let gain = [1.2, -0.7, 0.5];
    let up = [0.4, -0.8, 0.3, 1.1, -0.2, 0.6];
    let (dx, dg, db) = backward(&x, &gain, &up, 3)?;
    for (group, grad) in [&dx, &dg, &db].iter().enumerate() {
        for (i, &a) in grad.iter().enumerate() {
            let mut xp = x;
            let mut gp = gain;
            let mut bp = [0.; 3];
            let buffer = match group {
                0 => &mut xp[..],
                1 => &mut gp[..],
                _ => &mut bp[..],
            };
            buffer[i] += 1e-3;
            let plus = norm(&xp, &gp, &bp)?
                .iter()
                .zip(up)
                .map(|(a, b)| a * b)
                .sum::<f32>();
            let buffer = match group {
                0 => &mut xp[..],
                1 => &mut gp[..],
                _ => &mut bp[..],
            };
            buffer[i] -= 2e-3;
            let minus = norm(&xp, &gp, &bp)?
                .iter()
                .zip(up)
                .map(|(a, b)| a * b)
                .sum::<f32>();
            ensure(
                (a - (plus - minus) / 2e-3).abs() < 2e-3,
                format!("goal: normalization backward group {group} element {i} omitted a path"),
            )?;
        }
    }
    println!(
        "36 goal passed: two-layer full forward, causal prefix and LayerNorm x/gain/bias gradients"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn baseline_and_solution() {
        let m = ::ch36::Decoder::new(super::config(), 36).unwrap();
        assert_eq!(
            super::forward_with(&m, &[1, 2], crate::ch36::block)
                .unwrap()
                .len(),
            16
        );
        super::check_with(
            crate::solutions::ch36::block,
            crate::solutions::ch36::norm_backward,
        )
        .unwrap();
        assert!(super::forward_with(&m, &[], crate::ch36::block).is_err());
    }
}
