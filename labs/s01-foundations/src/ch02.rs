//! Learner: replace the numerical gradient with one analytical data pass.
use crate::scalar::{close, loss, Model};
#[derive(Debug)]
pub struct Work {
    pub gradient: Model,
    pub row_visits: usize,
}
pub fn gradient(model: Model, data: &[(f64, f64)]) -> Result<Work, String> {
    Ok(Work {
        gradient: crate::solutions::ch01::numerical_gradient(model, data)?,
        row_visits: 4 * data.len(),
    })
}
pub type GradientFn = fn(Model, &[(f64, f64)]) -> Result<Work, String>;
pub fn train(
    data: &[(f64, f64)],
    steps: usize,
    rate: f64,
    gradient: GradientFn,
) -> Result<(Model, usize), String> {
    crate::scalar::settings(rate)?;
    let mut model = [0., 0.];
    let mut visits = 0;
    loss(model, data)?;
    for _ in 0..steps {
        let work = gradient(model, data)?;
        visits += work.row_visits;
        for (p, g) in model.iter_mut().zip(work.gradient) {
            *p -= rate * g;
        }
        loss(model, data)?;
    }
    Ok((model, visits))
}
pub fn report(gradient: GradientFn, args: &[String]) -> Result<(), String> {
    let o = crate::args::Options::parse(args, &["--rate", "--steps", "--variant"])?;
    let data = crate::ch01::data_for(o.text("--variant", "calibration"))?;
    let work = gradient([0., 0.], &data)?;
    println!(
        "gradient={:?}; gradient row visits={} (reporting/validation excluded)",
        work.gradient, work.row_visits
    );
    let (model, visits) = train(
        &data,
        o.count("--steps", 100)?,
        o.number("--rate", 0.1)?,
        gradient,
    )?;
    println!(
        "trained={model:?}, MSE={:.10}; gradient row visits={visits}",
        loss(model, &data)?
    );
    Ok(())
}
pub fn verify(gradient: GradientFn) -> Result<(), String> {
    let data = [(0., 1.), (1., 3.), (3., 7.)];
    for m in [[0., 0.], [1.3, -0.4], [-0.7, 2.]] {
        let work = gradient(m, &data)?;
        let numeric = crate::solutions::ch01::numerical_gradient(m, &data)?;
        println!(
            "model={m:?}: analytical={:?}, numerical={numeric:?}, row visits={}",
            work.gradient, work.row_visits
        );
        if !work
            .gradient
            .into_iter()
            .zip(numeric)
            .all(|(a, b)| close(a, b))
        {
            return Err(
                "GOAL_NOT_MET: derivative disagreement: check signs, factor two, and mean reduction".into(),
            );
        }
        if work.row_visits != data.len() {
            return Err(
                "GOAL_NOT_MET: compute both slopes in one pass; count rows actually visited".into(),
            );
        }
    }
    let (model, _) = train(&data, 500, 0.03, gradient)?;
    if (crate::scalar::predict(model, 2.5) - 6.).abs() > 1e-3 {
        return Err("GOAL_NOT_MET: unfamiliar prediction failed".into());
    }
    Ok(())
}
pub fn run(args: &[String]) -> Result<(), String> {
    report(gradient, args)
}
pub fn check() -> Result<(), String> {
    verify(gradient)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::TRAIN;
    #[test]
    fn gradients_and_real_baseline() -> Result<(), String> {
        assert!(close(gradient([0., 0.], &TRAIN)?.gradient[0], -8.));
        assert!(gradient([0., 0.], &[]).is_err());
        crate::solutions::ch02::check()
    }
}
