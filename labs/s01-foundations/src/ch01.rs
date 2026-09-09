//! Learner: replace the whole candidate search with a numerical-gradient trainer.
//! Keep predict/loss and the command plumbing. See the lesson for the four probes.
use crate::scalar::{loss, predict, settings, Model, TRAIN};

pub const CANDIDATES: [Model; 3] = [[1., 0.], [1., 1.], [1.5, 0.5]];
pub fn candidate_search(data: &[(f64, f64)], candidates: &[Model]) -> Result<Model, String> {
    let mut best = *candidates.first().ok_or("provide at least one candidate")?;
    loss(best, data)?;
    for &candidate in candidates {
        if loss(candidate, data)? < loss(best, data)? {
            best = candidate;
        }
    }
    Ok(best)
}
pub fn train(data: &[(f64, f64)], steps: usize, rate: f64) -> Result<Model, String> {
    settings(rate)?;
    // This baseline searches fixed rules. Training step/rate controls affect your replacement.
    let _ = steps;
    candidate_search(data, &CANDIDATES)
}
pub fn data_for(variant: &str) -> Result<Vec<(f64, f64)>, String> {
    let mut data = TRAIN.to_vec();
    match variant {
        "calibration" => {}
        "alternate" => {
            for (x, y) in &mut data {
                *y = -0.5 * *x + 3.;
            }
        }
        "outlier" => {
            data[4].1 += 3.;
        }
        _ => return Err("variant must be calibration, alternate, or outlier".into()),
    }
    Ok(data)
}
pub fn run(args: &[String]) -> Result<(), String> {
    report(train, args)
}
pub fn check() -> Result<(), String> {
    verify(train)
}
pub type Trainer = fn(&[(f64, f64)], usize, f64) -> Result<Model, String>;
pub fn report(train: Trainer, args: &[String]) -> Result<(), String> {
    let options = crate::args::Options::parse(args, &["--rate", "--steps", "--variant"])?;
    let rate = options.number("--rate", 0.1)?;
    let steps = options.count("--steps", 100)?;
    let data = data_for(options.text("--variant", "calibration"))?;
    println!(
        "calibration pairs: {data:?}; zero-rule MSE={:.6}",
        loss([0., 0.], &data)?
    );
    for candidate in CANDIDATES {
        println!(
            "candidate weight={}, bias={}: MSE={:.6}",
            candidate[0],
            candidate[1],
            loss(candidate, &data)?
        );
    }
    println!(
        "comparison winner={:?}; requested numerical-training steps={steps}, rate={rate}",
        candidate_search(&data, &CANDIDATES)?
    );
    let model = train(&data, steps, rate)?;
    println!(
        "weight={:.6}, bias={:.6}; training MSE={:.9}",
        model[0],
        model[1],
        loss(model, &data)?
    );
    for x in [-1.5, 0.5, 1.5, 3.] {
        println!("unseen x={x}: prediction={:.6}", predict(model, x));
    }
    println!("The supplied candidate baseline ignores step/rate values; your numerical trainer uses them. Baseline success is not the learning goal.");
    Ok(())
}
pub fn verify(train: Trainer) -> Result<(), String> {
    let alternate = [(-2., 4.), (-1., 3.5), (0., 3.), (1., 2.5), (2., 2.)];
    for (name, data, x, target) in [
        ("calibration", &TRAIN[..], 0.5, 2.),
        ("changed line", &alternate[..], 1.5, 2.25),
    ] {
        let model = train(data, 150, 0.1)?;
        let mse = loss(model, data)?;
        let error = (predict(model, x) - target).abs();
        println!("{name}: MSE={mse:.9}, unseen absolute error={error:.9}");
        if mse > 1e-8 || error > 1e-4 {
            return Err("GOAL_NOT_MET: learn parameters from both supplied linear fixtures with numerical slopes; see ch01.rs".into());
        }
    }
    // Asymmetry exposes computing the bias slope after changing the weight.
    let first = train(&[(1., 3.), (2., 5.)], 1, 0.1)?;
    if !crate::scalar::close(first[0], 1.3) || !crate::scalar::close(first[1], 0.8) {
        return Err(format!(
            "GOAL_NOT_MET: one simultaneous step expected [1.3,0.8], got {first:?}"
        ));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_is_useful_and_boundaries_hold() -> Result<(), String> {
        assert_eq!(loss([0., 0.], &TRAIN)?, 9.);
        assert_eq!(
            candidate_search(&TRAIN, &[[1., 0.], [2., 1.], [1., 1.]])?,
            [2., 1.]
        );
        assert!(candidate_search(&TRAIN, &[]).is_err());
        assert!(loss(train(&TRAIN, 100, 0.1)?, &TRAIN)? <= 0.75 + 1e-8);
        assert!(train(&[], 1, 0.1).is_err());
        assert!(train(&TRAIN, 0, f64::NAN).is_err());
        assert!(loss([0., 0.], &[(f64::INFINITY, 1.)]).is_err());
        crate::solutions::ch01::check()
    }
}
