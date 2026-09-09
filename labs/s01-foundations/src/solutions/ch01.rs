//! Explained solution: each gradient component probes the SAME old parameters.
//! Average squared error is used without a factor of one half.
use crate::scalar::{loss, settings, Model};
pub fn numerical_gradient(model: Model, data: &[(f64, f64)]) -> Result<Model, String> {
    let h = 1e-5;
    let mut gradient = [0.; 2];
    for (j, g) in gradient.iter_mut().enumerate() {
        let mut plus = model;
        let mut minus = model;
        plus[j] += h;
        minus[j] -= h;
        *g = (loss(plus, data)? - loss(minus, data)?) / (2. * h);
    }
    Ok(gradient)
}
pub fn train(data: &[(f64, f64)], steps: usize, rate: f64) -> Result<Model, String> {
    settings(rate)?;
    let mut model = [0., 0.];
    loss(model, data)?;
    for _ in 0..steps {
        let gradient = numerical_gradient(model, data)?;
        // Store both slopes before mutation: one full-batch update.
        for (p, g) in model.iter_mut().zip(gradient) {
            *p -= rate * g;
        }
        loss(model, data)?;
    }
    Ok(model)
}
pub fn run(args: &[String]) -> Result<(), String> {
    crate::ch01::report(train, args)
}
pub fn check() -> Result<(), String> {
    crate::ch01::verify(train)
}
