//! The derivative of e² is 2e; prediction derivatives are x and 1.
use crate::{
    ch02::Work,
    scalar::{predict, Model},
};
pub fn gradient(model: Model, data: &[(f64, f64)]) -> Result<Work, String> {
    if data.is_empty() || model.iter().any(|v| !v.is_finite()) {
        return Err("gradient requires nonempty data and finite parameters".into());
    }
    let mut gradient = [0., 0.];
    let mut row_visits = 0;
    for &(x, y) in data {
        if !x.is_finite() || !y.is_finite() {
            return Err("gradient examples must be finite".into());
        }
        let error = predict(model, x) - y;
        gradient[0] += 2. * error * x / data.len() as f64;
        gradient[1] += 2. * error / data.len() as f64;
        row_visits += 1;
    }
    if gradient.iter().any(|g| !g.is_finite()) {
        return Err("gradient overflow".into());
    }
    Ok(Work {
        gradient,
        row_visits,
    })
}
pub fn run(args: &[String]) -> Result<(), String> {
    crate::ch02::report(gradient, args)
}
pub fn check() -> Result<(), String> {
    crate::ch02::verify(gradient)
}
