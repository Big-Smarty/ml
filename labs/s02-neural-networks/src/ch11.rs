//! Learner: complete hidden-layer backpropagation, then replace SGD with momentum and Adam.
//! Initialization and checkpoint experiments use supplied Mlp::new/save/restore plumbing.
use crate::{
    data::{self, Dataset, CLASSES},
    linear::evaluate,
    mlp::{restore, save, Kind, Mlp, Optimizer},
};
pub type Derivative = fn(&Mlp, &Dataset, usize, usize) -> Result<(f64, Vec<f64>), String>;
pub type Update = fn(&mut Optimizer, &mut [f64], &[f64]) -> Result<(), String>;
pub fn gradient(
    model: &Mlp,
    data: &Dataset,
    start: usize,
    end: usize,
) -> Result<(f64, Vec<f64>), String> {
    if data.in_features() != model.input || start >= end || end > data.len() {
        return Err("invalid MLP batch".into());
    }
    let mut gradient = vec![0.; model.parameters.len()];
    let mut loss = 0.;
    let (_, _, w2, b2) = model.parameter_offsets();
    // Working random-feature classifier: hidden units are frozen; fit output weights.
    for n in start..end {
        let (hidden, z) = model.forward(data.image(n));
        let (row_loss, mut p) = crate::linear::objective(z, data.labels[n] as usize)?;
        loss += row_loss;
        p[data.labels[n] as usize] -= 1.;
        for c in 0..CLASSES {
            gradient[b2 + c] += p[c];
            for (j, &h) in hidden.iter().enumerate() {
                gradient[w2 + c * model.hidden + j] += p[c] * h;
            }
        }
    }
    for g in &mut gradient {
        *g /= (end - start) as f64;
    }
    Ok((loss / (end - start) as f64, gradient))
}
pub fn update(
    optimizer: &mut Optimizer,
    parameters: &mut [f64],
    gradient: &[f64],
) -> Result<(), String> {
    optimizer.validate(parameters, gradient)?;
    // Plain SGD baseline; stored optimizer kind is the goal to implement next.
    optimizer.step_count = optimizer
        .step_count
        .checked_add(1)
        .ok_or("optimizer step overflow")?;
    for (p, &g) in parameters.iter_mut().zip(gradient) {
        *p -= optimizer.learning_rate * g;
    }
    if parameters.iter().any(|v| !v.is_finite()) {
        return Err("SGD parameter overflow".into());
    }
    Ok(())
}
pub fn train(
    model: &mut Mlp,
    optimizer: &mut Optimizer,
    data: &Dataset,
    epochs: usize,
    batch: usize,
    derivative: Derivative,
    update: Update,
) -> Result<(), String> {
    if epochs == 0 || batch == 0 {
        return Err("epochs and batch must be positive".into());
    }
    for _ in 0..epochs {
        for start in (0..data.len()).step_by(batch) {
            let (_, g) = derivative(
                model,
                data,
                start,
                start.saturating_add(batch).min(data.len()),
            )?;
            update(optimizer, &mut model.parameters, &g)?;
        }
    }
    model.metrics(data)?;
    Ok(())
}
pub fn run(args: &[String]) -> Result<(), String> {
    report(args, gradient, update)
}
pub fn check() -> Result<(), String> {
    verify(gradient, update)
}
pub fn report(args: &[String], derivative: Derivative, update: Update) -> Result<(), String> {
    let (mut kind, mut initialization) = (Kind::Adam, "scaled");
    let mut rest = args;
    while let [flag, value, tail @ ..] = rest {
        match flag.as_str() {
            "--optimizer" => {
                kind = match value.as_str() {
                    "adam" => Kind::Adam,
                    "momentum" => Kind::Momentum,
                    _ => return Err("optimizer must be adam or momentum".into()),
                }
            }
            "--init" => {
                initialization = match value.as_str() {
                    "scaled" => "scaled",
                    "zero" => "zero",
                    "large" => "large",
                    _ => return Err("initialization must be scaled, zero or large".into()),
                }
            }
            _ => break,
        }
        rest = tail;
    }
    let args = rest;
    let (fit, held, epochs) = data::datasets(args)?;
    data::inspect(&fit);
    let mut model = Mlp::new(fit.in_features(), 16)?;
    match initialization {
        "zero" => model.parameters.fill(0.),
        "large" => model.parameters.iter_mut().for_each(|w| *w *= 20.),
        _ => {}
    }
    println!("initialization={initialization}");
    let mut optimizer = Optimizer::new(kind, model.parameters.len());
    let (_, initial_gradient) = derivative(&model, &fit, 0, fit.len().min(10))?;
    let (_, bias1, _, _) = model.parameter_offsets();
    let first = &initial_gradient[..bias1];
    let max_abs = first.iter().map(|v| v.abs()).fold(0., f64::max);
    let zeros = first.iter().filter(|&&v| v == 0.).count();
    println!(
        "first-layer weight gradient before training: max |g|={max_abs:.9}, zeros={zeros}/{}",
        first.len()
    );
    let before = model.metrics(&held)?;
    train(
        &mut model,
        &mut optimizer,
        &fit,
        epochs,
        10,
        derivative,
        update,
    )?;
    let after = evaluate(&held, |x| Ok(model.forward(x).1), true)?;
    let mut inactive = vec![0usize; model.hidden];
    for n in 0..held.len() {
        for (count, h) in inactive.iter_mut().zip(model.forward(held.image(n)).0) {
            *count += usize::from(h == 0.);
        }
    }
    println!("requested optimizer={kind:?}; learner starts as SGD, solution implements requested algorithm");
    println!("fit loss={:.6}; held-out loss {:.6}->{:.6}, accuracy {:.3}->{:.3}; zero-activation counts / {} = {:?}",model.metrics(&fit)?.0,before.0,after.loss,before.1,after.accuracy,held.len(),inactive);
    resume_check(&model, &optimizer, &fit, derivative, update)?;
    Ok(())
}
fn resume_check(
    model: &Mlp,
    optimizer: &Optimizer,
    data: &Dataset,
    derivative: Derivative,
    update: Update,
) -> Result<(), String> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("s02-resume-{}-{stamp}", std::process::id()));
    std::fs::create_dir(&dir).map_err(|e| e.to_string())?;
    let path = dir.join("model.bin");
    let result = (|| {
        save(&path, model, optimizer)?;
        let (mut restored, mut restored_optimizer) = restore(&path)?;
        if restored.parameters != model.parameters
            || restored_optimizer.m != optimizer.m
            || restored_optimizer.v != optimizer.v
        {
            return Err("checkpoint changed model or moment buffers".into());
        }
        let mut next = model.clone();
        let mut next_optimizer = optimizer.clone();
        let (_, g) = derivative(model, data, 0, data.len().min(7))?;
        update(&mut next_optimizer, &mut next.parameters, &g)?;
        update(&mut restored_optimizer, &mut restored.parameters, &g)?;
        if next.parameters != restored.parameters
            || next_optimizer.step_count != restored_optimizer.step_count
        {
            return Err("next update differs after resume".into());
        }
        println!(
            "checkpoint restored step {}; identical next update on the same seven rows",
            optimizer.step_count
        );
        // Corrupt input must fail; only the newly-created scratch file is changed.
        std::fs::write(&path, b"bad checkpoint").map_err(|e| e.to_string())?;
        if restore(&path).is_ok() {
            return Err("corrupt checkpoint accepted".into());
        }
        Ok(())
    })();
    std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
    result
}
pub fn verify(derivative: Derivative, update: Update) -> Result<(), String> {
    // Smooth, unfamiliar signed inputs, chosen away from every ReLU zero.
    let data = Dataset {
        images: vec![0.2, -0.7, 0.4, 0.3, -0.2, 0.8],
        labels: vec![2, 7],
        rows: 1,
        cols: 3,
    };
    let model = Mlp::new(3, 4)?;
    let (_, g) = derivative(&model, &data, 0, 2)?;
    for (j, &actual) in g.iter().enumerate() {
        let (mut plus, mut minus) = (model.clone(), model.clone());
        plus.parameters[j] += 1e-5;
        minus.parameters[j] -= 1e-5;
        let numeric = (plus.metrics(&data)?.0 - minus.metrics(&data)?.0) / 2e-5;
        if !crate::close(actual, numeric) {
            return Err(format!("GOAL_NOT_MET: MLP parameter {j}: gradient {:.9}, numeric {numeric:.9}. Complete the hidden gradient before optimizer experiments.",g[j]));
        }
    }
    println!("all 66 MLP gradients agree on the signed-input transfer batch");
    for kind in [Kind::Momentum, Kind::Adam] {
        let mut optimizer = Optimizer::new(kind, 1);
        optimizer.learning_rate = 0.1;
        let mut parameter = [1.];
        update(&mut optimizer, &mut parameter, &[2.])?;
        update(&mut optimizer, &mut parameter, &[0.])?;
        let expected = if kind == Kind::Momentum {
            0.62
        } else {
            let first = 1. - 0.1 * 2. / (2. + 1e-8);
            first
                - 0.1 * (0.18 / (1. - 0.9f64.powi(2)))
                    / ((0.003996 / (1. - 0.999f64.powi(2))).sqrt() + 1e-8)
        };
        if !crate::close(parameter[0], expected)
            || optimizer.step_count != 2
            || optimizer.m[0] == 0.
        {
            return Err(format!("GOAL_NOT_MET: {kind:?} two-step parameter {}, expected {expected}; retain moments and count once per vector",parameter[0]));
        }
        println!("{kind:?}: independent two-step scalar update agrees");
        let fit = data::fixture(true)?;
        let held = data::fixture(false)?;
        let mut model = Mlp::new(fit.in_features(), 16)?;
        let mut optimizer = Optimizer::new(kind, model.parameters.len());
        train(
            &mut model,
            &mut optimizer,
            &fit,
            160,
            10,
            derivative,
            update,
        )?;
        let a = model.metrics(&fit)?;
        let b = model.metrics(&held)?;
        println!(
            "{kind:?}: fit loss={:.6}, held-out loss={:.6}, accuracy={:.3}",
            a.0, b.0, b.1
        );
        if a.0 > 0.2 || b.1 < 0.75 {
            return Err("GOAL_NOT_MET: MLP goal: low training loss and at least 9/12 held-out block glyph decisions; inspect gradients and active units".into());
        }
        resume_check(&model, &optimizer, &fit, derivative, update)?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frozen_feature_baseline_and_full_mlp() -> Result<(), String> {
        let fit = data::fixture(true)?;
        let mut m = Mlp::new(15, 16)?;
        let before = m.metrics(&fit)?.0;
        let mut o = Optimizer::new(Kind::Momentum, m.parameters.len());
        train(&mut m, &mut o, &fit, 20, 10, gradient, update)?;
        assert!(m.metrics(&fit)?.0 < before);
        assert!(Mlp::new(0, 4).is_err());
        let mut p = [1.];
        let mut o = Optimizer::new(Kind::Adam, 1);
        assert!(update(&mut o, &mut p, &[f64::NAN]).is_err());
        crate::solutions::ch11::check()
    }
    #[test]
    fn zero_initialization_freezes_hidden_and_oversized_values_fail() -> Result<(), String> {
        let fit = data::fixture(true)?;
        let mut m = Mlp::new(15, 4)?;
        m.parameters.fill(0.);
        let (_, g) = crate::solutions::ch11::gradient(&m, &fit, 0, fit.len())?;
        let (_, _, _, b2) = m.parameter_offsets();
        assert!(g[..b2].iter().all(|&v| v == 0.));
        m.parameters.fill(f64::MAX);
        assert!(m.metrics(&fit).is_err());
        Ok(())
    }
}
