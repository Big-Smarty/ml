//! Each logit derivative owns one class row; multiply by each input feature.
use crate::{
    ch10::Gradient,
    data::{Dataset, CLASSES},
    linear::{objective, Linear},
};
pub fn gradient(
    model: &Linear,
    data: &Dataset,
    start: usize,
    end: usize,
) -> Result<Gradient, String> {
    if data.in_features() != model.input || start >= end || end > data.len() {
        return Err("invalid linear minibatch".into());
    }
    let mut result = Gradient {
        weights: vec![0.; model.weights.len()],
        bias: [0.; CLASSES],
    };
    for n in start..end {
        let image = data.image(n);
        let (_, mut p) = objective(model.logits(image)?, data.labels[n] as usize)?;
        p[data.labels[n] as usize] -= 1.;
        for (class, &signal) in p.iter().enumerate() {
            result.bias[class] += signal / (end - start) as f64;
            for (feature, &x) in image.iter().enumerate() {
                result.weights[class * model.input + feature] += signal * x / (end - start) as f64;
            }
        }
    }
    Ok(result)
}
pub fn per_class_recall(confusion: &crate::linear::Confusion) -> [Option<f64>; CLASSES] {
    std::array::from_fn(|class| {
        let count = confusion[class].iter().sum::<usize>();
        (count > 0).then(|| confusion[class][class] as f64 / count as f64)
    })
}
pub fn run(args: &[String]) -> Result<(), String> {
    crate::ch10::report(args, gradient, per_class_recall)
}
pub fn check() -> Result<(), String> {
    crate::ch10::verify(gradient, per_class_recall)
}
