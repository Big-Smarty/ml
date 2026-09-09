//! Learner: build an entire logistic trainer to replace a constant prior model.
use crate::classifier::{self, Model, Row, TRAIN};
pub fn train(data: &[Row], steps: usize, rate: f64) -> Result<Model, String> {
    classifier::validate(data)?;
    crate::scalar::settings(rate)?;
    // Laplace smoothing supplies a finite, useful class-prior baseline.
    let positives = data.iter().filter(|(_, y)| *y).count() as f64;
    let p = (positives + 1.) / (data.len() as f64 + 2.);
    let _ = steps;
    Ok([0., 0., (p / (1. - p)).ln()])
}
pub type Trainer = fn(&[Row], usize, f64) -> Result<Model, String>;
pub fn report(train: Trainer, args: &[String]) -> Result<(), String> {
    let o = crate::args::Options::parse(args, &["--rate", "--steps", "--variant", "--threshold"])?;
    let threshold = o.number("--threshold", 0.5)?;
    if !(0.0..=1.0).contains(&threshold) {
        return Err("threshold must be in[0,1]".into());
    }
    let mut data = TRAIN.to_vec();
    match o.text("--variant", "original") {
        "original" => {}
        "reversed" => {
            for (_, y) in &mut data {
                *y = !*y;
            }
        }
        "contradiction" => {
            let (x, y) = data[0];
            data.push((x, !y));
        }
        _ => return Err("variant must be original, reversed, or contradiction".into()),
    }
    let m = train(&data, o.count("--steps", 800)?, o.number("--rate", 0.2)?)?;
    println!(
        "fault classifier: parameters={m:?}, mean BCE={:.6}",
        classifier::loss(m, &data)?
    );
    for x in [[-1., -0.5], [0.1, 0.1], [1.2, 0.7]] {
        let z = classifier::logit(m, x);
        let p = classifier::sigmoid(z);
        println!(
            "{x:?}: logit={z:.4}, probability={p:.4}, class={}",
            u8::from(p >= threshold)
        );
    }
    for (z, y) in [(1000., false), (-1000., true), (1000., true)] {
        println!(
            "extreme logit={z}, target={y}: BCE={}",
            classifier::bce(z, y)?
        );
    }
    Ok(())
}
pub fn verify(train: Trainer) -> Result<(), String> {
    let m = train(&TRAIN, 800, 0.2)?;
    let l = classifier::loss(m, &TRAIN)?;
    println!("training BCE={l:.6} (goal <0.04)");
    if l >= 0.04 {
        return Err(
            "GOAL_NOT_MET: implement mean BCE gradients and repeated simultaneous updates".into(),
        );
    }
    let reversed: Vec<_> = TRAIN.iter().map(|&(x, y)| (x, !y)).collect();
    let r = train(&reversed, 800, 0.2)?;
    for (x, y) in [([-1., -0.5], false), ([1.2, 0.7], true)] {
        if (classifier::sigmoid(classifier::logit(m, x)) >= 0.5) != y
            || (classifier::sigmoid(classifier::logit(r, x)) >= 0.5) == y
        {
            return Err("GOAL_NOT_MET: unfamiliar input/reversed labels failed".into());
        }
    }
    let one = train(&[([2., 1.], true)], 1, 0.1)?;
    if !one
        .into_iter()
        .zip([0.1, 0.05, 0.05])
        .all(|(a, b)| crate::scalar::close(a, b))
    {
        return Err(format!(
            "GOAL_NOT_MET: one step expected [0.1,0.05,0.05], got {one:?}"
        ));
    }
    Ok(())
}
pub fn run(args: &[String]) -> Result<(), String> {
    report(train, args)
}
pub fn check() -> Result<(), String> {
    verify(train)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_and_stability() -> Result<(), String> {
        assert!(
            classifier::loss(train(&TRAIN, 800, 0.2)?, &TRAIN)? <= std::f64::consts::LN_2 + 1e-10
        );
        assert_eq!(classifier::bce(1000., false)?, 1000.);
        assert_eq!(classifier::bce(-1000., true)?, 1000.);
        assert!(train(&[], 1, 0.1).is_err());
        assert!(classifier::bce(f64::NAN, true).is_err());
        crate::solutions::ch04::check()
    }
}
