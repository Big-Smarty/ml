//! Supplied two-feature classifier plumbing, shared with the evaluation chapter.
pub type Row = ([f64; 2], bool);
pub type Model = [f64; 3];
pub const TRAIN: [Row; 8] = [
    ([-2., -1.], false),
    ([-1.5, 0.2], false),
    ([-0.8, -1.3], false),
    ([-0.2, -0.7], false),
    ([0.4, 0.8], true),
    ([0.9, 1.5], true),
    ([1.4, 0.1], true),
    ([2., 1.], true),
];
pub fn logit(m: Model, x: [f64; 2]) -> f64 {
    m[0] * x[0] + m[1] * x[1] + m[2]
}
pub fn sigmoid(z: f64) -> f64 {
    if z >= 0. {
        1. / (1. + (-z).exp())
    } else {
        let e = z.exp();
        e / (1. + e)
    }
}
pub fn validate(data: &[Row]) -> Result<(), String> {
    if data.is_empty() || data.iter().any(|(x, _)| x.iter().any(|v| !v.is_finite())) {
        Err("classifier requires nonempty finite two-feature rows".into())
    } else {
        Ok(())
    }
}
pub fn bce(z: f64, y: bool) -> Result<f64, String> {
    if !z.is_finite() {
        return Err("logit must be finite".into());
    }
    Ok(z.max(0.) - z * f64::from(y) + (-z.abs()).exp().ln_1p())
}
pub fn loss(m: Model, data: &[Row]) -> Result<f64, String> {
    validate(data)?;
    let l = data
        .iter()
        .map(|&(x, y)| bce(logit(m, x), y))
        .sum::<Result<f64, _>>()?
        / data.len() as f64;
    if l.is_finite() {
        Ok(l)
    } else {
        Err("BCE overflow".into())
    }
}
