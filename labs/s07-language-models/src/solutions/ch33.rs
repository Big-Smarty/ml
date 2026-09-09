//! The same logit derivative feeds bias, output outer product and selected embedding row.
//! Accumulate using old parameters; mutate only after every pair has contributed.
use crate::{
    byte_model::{self, Model},
    LabResult,
};
pub fn count_loss(train: &[u8], data: &[u8]) -> LabResult<f64> {
    crate::ensure(
        train.len() >= 2 && data.len() >= 2,
        "count model needs two bytes in train and evaluation",
    )?;
    let counts = byte_model::byte_bigrams(train);
    Ok(data
        .windows(2)
        .map(|pair| {
            let row = counts
                .iter()
                .filter(|((x, _), _)| *x == pair[0])
                .map(|(_, n)| n)
                .sum::<usize>();
            let observed = counts.get(&(pair[0], pair[1])).copied().unwrap_or(0);
            -((observed + 1) as f64 / (row + 256) as f64).ln()
        })
        .sum::<f64>()
        / (data.len() - 1) as f64)
}
pub fn update(model: &mut Model, data: &[u8], rate: f64) -> LabResult {
    byte_model::validate(model, data, rate)?;
    let mut de = vec![0.0; model.embeddings.len()];
    let mut dw = vec![0.0; model.output_weights.len()];
    let mut db = vec![0.0; 256];
    for pair in data.windows(2) {
        let p = model.probabilities(pair[0]);
        for class in 0..256 {
            let g = (p[class] - f64::from(class == pair[1] as usize)) / (data.len() - 1) as f64;
            db[class] += g;
            for h in 0..model.width {
                let ei = pair[0] as usize * model.width + h;
                let wi = h * 256 + class;
                de[ei] += g * model.output_weights[wi];
                dw[wi] += g * model.embeddings[ei];
            }
        }
    }
    for (params, grads) in [
        (&mut model.embeddings, de),
        (&mut model.output_weights, dw),
        (&mut model.bias, db),
    ] {
        for (p, g) in params.iter_mut().zip(grads) {
            *p -= rate * g;
        }
    }
    Ok(())
}
pub fn run(args: &[String]) -> LabResult {
    byte_model::run_with(args, update, count_loss)
}
pub fn check() -> LabResult {
    byte_model::check_with(update, count_loss)
}
