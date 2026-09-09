//! Learner: replace bias-only fitting with the complete mean multiclass gradient.
use crate::{
    data::{self, Dataset, CLASSES},
    linear::{evaluate, objective, Confusion, Linear},
};
#[derive(Debug)]
pub struct Gradient {
    pub weights: Vec<f64>,
    pub bias: [f64; CLASSES],
}
pub type Derivative = fn(&Linear, &Dataset, usize, usize) -> Result<Gradient, String>;
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
    // Working intercept-only trainer. Class frequencies can be learned without pixels.
    for n in start..end {
        let (_, mut p) = objective(model.logits(data.image(n))?, data.labels[n] as usize)?;
        p[data.labels[n] as usize] -= 1.;
        for (g, v) in result.bias.iter_mut().zip(p) {
            *g += v / (end - start) as f64;
        }
    }
    Ok(result)
}
pub fn train(
    model: &mut Linear,
    data: &Dataset,
    epochs: usize,
    batch: usize,
    rate: f64,
    derivative: Derivative,
) -> Result<(), String> {
    if epochs == 0 || batch == 0 || !rate.is_finite() || rate <= 0. {
        return Err("epochs, batch, and finite learning rate must be positive".into());
    }
    for _ in 0..epochs {
        for start in (0..data.len()).step_by(batch) {
            let g = derivative(
                model,
                data,
                start,
                start.saturating_add(batch).min(data.len()),
            )?;
            if g.weights.len() != model.weights.len()
                || g.weights.iter().chain(&g.bias).any(|v| !v.is_finite())
            {
                return Err("invalid gradient shape or values".into());
            }
            for (w, g) in model.weights.iter_mut().zip(g.weights) {
                *w -= rate * g;
            }
            for (b, g) in model.bias.iter_mut().zip(g.bias) {
                *b -= rate * g;
            }
        }
    }
    evaluate(data, |x| model.logits(x), false)?;
    Ok(())
}
pub type Recall = fn(&Confusion) -> [Option<f64>; CLASSES];
/// Baseline reports aggregate accuracy only; implement the ten per-class recalls.
pub fn per_class_recall(_confusion: &Confusion) -> [Option<f64>; CLASSES] {
    [None; CLASSES]
}
pub fn run(args: &[String]) -> Result<(), String> {
    report(args, gradient, per_class_recall)
}
pub fn check() -> Result<(), String> {
    verify(gradient, per_class_recall)
}
pub fn report(args: &[String], derivative: Derivative, recall: Recall) -> Result<(), String> {
    let (brightness, args) = match args {
        [flag, value, tail @ ..] if flag == "--brightness" => (
            value
                .parse::<f64>()
                .map_err(|_| "brightness must be a number")?,
            tail,
        ),
        _ => (1., args),
    };
    if !brightness.is_finite() || !(0.0..=2.0).contains(&brightness) {
        return Err("brightness must be finite in 0..=2".into());
    }
    let (fit, mut held, epochs) = data::datasets(args)?;
    for pixel in &mut held.images {
        *pixel *= brightness;
    }
    println!("held-out brightness multiplier={brightness}; fit data unchanged");
    data::inspect(&fit);
    let mut model = Linear::new(fit.in_features())?;
    let before = evaluate(&held, |x| model.logits(x), false)?;
    train(&mut model, &fit, epochs, 10, 0.25, derivative)?;
    let training = evaluate(&fit, |x| model.logits(x), false)?;
    let after = evaluate(&held, |x| model.logits(x), true)?;
    println!(
        "fit loss={:.6} accuracy={:.3}; held-out loss {:.6}->{:.6}, accuracy {:.3}->{:.3}",
        training.loss, training.accuracy, before.loss, after.loss, before.accuracy, after.accuracy
    );
    println!("per-class recall {:?}; baseline leaves these unreported until per_class_recall is implemented",recall(&after.confusion));
    if args.is_empty() {
        println!("Fixture: course block glyphs plus two ambiguous blanks; these are not MNIST accuracy results.");
    }
    Ok(())
}
pub fn verify(derivative: Derivative, recall: Recall) -> Result<(), String> {
    let data = data::fixture(true)?;
    let model = Linear::new(data.in_features())?;
    let g = derivative(&model, &data, 1, 8)?;
    let score = |m: &Linear| -> Result<f64, String> {
        let mut sum = 0.;
        for n in 1..8 {
            sum += objective(m.logits(data.image(n))?, data.labels[n] as usize)?.0;
        }
        Ok(sum / 7.)
    };
    for j in [0, 16, 39, 100, 149, 150, 159] {
        let (mut plus, mut minus) = (model.clone(), model.clone());
        let nw = model.weights.len();
        let actual = if j < nw {
            plus.weights[j] += 1e-5;
            minus.weights[j] -= 1e-5;
            g.weights[j]
        } else {
            plus.bias[j - nw] += 1e-5;
            minus.bias[j - nw] -= 1e-5;
            g.bias[j - nw]
        };
        let numeric = (score(&plus)? - score(&minus)?) / 2e-5;
        if !crate::close(actual, numeric) {
            return Err(format!("GOAL_NOT_MET: linear parameter {j}: gradient {actual:.8}, expected central difference {numeric:.8}; implement pixel-to-class weight gradients"));
        }
    }
    for rotate in [0, 3] {
        let mut fit = data.clone();
        for y in &mut fit.labels {
            *y = (*y + rotate) % 10;
        }
        let mut model = Linear::new(fit.in_features())?;
        train(&mut model, &fit, 180, 7, 0.25, derivative)?;
        let metrics = evaluate(&fit, |x| model.logits(x), false)?;
        println!(
            "label rotation={rotate}: fit loss={:.6}, accuracy={:.3}",
            metrics.loss, metrics.accuracy
        );
        if metrics.loss > 0.25 || metrics.accuracy < 0.9 {
            return Err(
                "GOAL_NOT_MET: goal: fit pixel features, including changed label mapping and a short final batch"
                    .into(),
            );
        }
    }
    println!("pixel-gradient/training checkpoint passed; checking per-class recall");
    let mut counts = [[0usize; CLASSES]; CLASSES];
    counts[1][1] = 1;
    counts[1][7] = 1;
    counts[7][7] = 2;
    let actual = recall(&counts);
    if actual[1] != Some(0.5)
        || actual[7] != Some(1.)
        || actual
            .iter()
            .enumerate()
            .any(|(c, v)| c != 1 && c != 7 && v.is_some())
    {
        return Err(format!("GOAL_NOT_MET: per-class recall needs class1=Some(0.5), class7=Some(1.0), absent classes=None; got {actual:?}"));
    }
    println!("per-class recall correctly distinguishes mistakes and absent classes");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_scoring_bias_baseline_and_full_trainer() -> Result<(), String> {
        let fit = data::fixture(true)?;
        let mut model = Linear::new(15)?;
        train(&mut model, &fit, 2, 7, 0.1, gradient)?;
        assert!(evaluate(&fit, |x| model.logits(x), false)?.loss < 10f64.ln() + 0.05);
        assert!((objective([1e16; 10], 3)?.0 - 10f64.ln()).abs() < 1e-12);
        assert!(objective([0.; 10], 10).is_err());
        let mut huge = [0.; 10];
        huge[1] = f64::MAX / 2.;
        assert!(evaluate(&fit, |_| Ok(huge), false).is_err());
        assert!(model.logits(&[f64::NAN; 15]).is_err());
        let z = [2., 1., 0., -1., -2., -3., -4., -5., -6., -7.];
        let (_, p) = objective(z, 1)?;
        for j in 0..10 {
            let (mut plus, mut minus) = (z, z);
            plus[j] += 1e-5;
            minus[j] -= 1e-5;
            let numeric = (objective(plus, 1)?.0 - objective(minus, 1)?.0) / 2e-5;
            assert!(crate::close(p[j] - f64::from(j == 1), numeric));
        }
        crate::solutions::ch10::check()
    }
}
