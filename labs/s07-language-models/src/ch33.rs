//! Learner: replace the unigram bias update with the embedding classifier's complete gradient.
use crate::{
    byte_model::{self, Model},
    LabResult,
};
pub fn count_loss(train: &[u8], data: &[u8]) -> LabResult<f64> {
    crate::ensure(
        train.len() >= 2 && data.len() >= 2,
        "count model needs two bytes in train and evaluation",
    )?;
    // Uniform byte predictor; replace with a smoothed conditional successor-count model.
    Ok(256_f64.ln())
}
pub fn update(model: &mut Model, data: &[u8], rate: f64) -> LabResult {
    byte_model::validate(model, data, rate)?;
    // Working baseline: learn output frequency, leaving representations fixed.
    let mut gradient = vec![0.0; 256];
    for pair in data.windows(2) {
        let probabilities = model.probabilities(pair[0]);
        for (class, g) in gradient.iter_mut().enumerate() {
            *g += (probabilities[class] - f64::from(class == pair[1] as usize))
                / (data.len() - 1) as f64;
        }
    }
    for (p, g) in model.bias.iter_mut().zip(gradient) {
        *p -= rate * g;
    }
    Ok(())
}
pub fn run(args: &[String]) -> LabResult {
    byte_model::run_with(args, update, count_loss)
}
pub fn check() -> LabResult {
    byte_model::check_with(update, count_loss)
}
