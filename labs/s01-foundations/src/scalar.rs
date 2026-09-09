//! Supplied arithmetic and input validation. Parameters are [weight, bias].
pub type Model = [f64; 2];
pub const TRAIN: [(f64, f64); 5] = [(-2., -3.), (-1., -1.), (0., 1.), (1., 3.), (2., 5.)];
pub fn predict(model: Model, x: f64) -> f64 {
    model[0] * x + model[1]
}
pub fn loss(model: Model, data: &[(f64, f64)]) -> Result<f64, String> {
    if data.is_empty()
        || model.iter().any(|v| !v.is_finite())
        || data.iter().any(|(x, y)| !x.is_finite() || !y.is_finite())
    {
        return Err("MSE needs nonempty finite data and finite parameters".into());
    }
    let value = data
        .iter()
        .map(|&(x, y)| (predict(model, x) - y).powi(2))
        .sum::<f64>()
        / data.len() as f64;
    if value.is_finite() {
        Ok(value)
    } else {
        Err("MSE overflow; reduce scale or step size".into())
    }
}
pub fn settings(rate: f64) -> Result<(), String> {
    if rate.is_finite() && rate > 0. {
        Ok(())
    } else {
        Err("learning rate must be finite and positive".into())
    }
}
pub fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-6 + 1e-4 * a.abs().max(b.abs())
}
