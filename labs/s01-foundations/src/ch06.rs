//! Learner: extend this full-batch SGD trainer with shuffled minibatches, L2 and saved-best early stopping.
use crate::scalar::{loss, Model};
pub const TRAIN: [(f64, f64); 12] = [
    (-3., -5.2),
    (-2.5, -3.7),
    (-2., -3.3),
    (-1.5, -1.7),
    (-1., -1.2),
    (-0.5, 0.4),
    (0., 0.8),
    (0.5, 2.2),
    (1., 2.7),
    (1.5, 4.4),
    (2., 4.8),
    (2.5, 6.4),
];
pub const VALIDATION: [(f64, f64); 6] = [
    (-2.75, -4.4),
    (-1.25, -1.4),
    (-0.25, 0.6),
    (0.75, 2.5),
    (1.75, 4.6),
    (2.75, 6.3),
];
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub epochs: usize,
    pub batch: usize,
    pub rate: f64,
    pub l2: f64,
    pub seed: u64,
    pub patience: usize,
}
pub fn defaults() -> Config {
    Config {
        epochs: 80,
        batch: 12,
        rate: 0.03,
        l2: 0.,
        seed: 7,
        patience: 0,
    }
}
#[derive(Debug)]
pub struct Run {
    pub model: Model,
    pub curve: Vec<(f64, f64)>,
    pub steps: usize,
}
pub fn validate(train: &[(f64, f64)], val: &[(f64, f64)], c: Config) -> Result<(), String> {
    loss([0.; 2], train)?;
    loss([0.; 2], val)?;
    crate::scalar::settings(c.rate)?;
    if c.batch == 0 || c.batch > train.len() || !c.l2.is_finite() || c.l2 < 0. {
        return Err("batch must be1..N and L2 finite/nonnegative".into());
    }
    Ok(())
}
pub fn fit(train: &[(f64, f64)], val: &[(f64, f64)], c: Config) -> Result<Run, String> {
    validate(train, val, c)?;
    if c.batch != train.len() || c.l2 != 0. || c.patience != 0 {
        return Err(
            "GOAL_NOT_MET: baseline supports full-batch SGD only; implement minibatches, L2 and saved-best early stopping".into(),
        );
    }
    let mut model = [0.; 2];
    let mut curve = vec![(loss(model, train)?, loss(model, val)?)];
    for _ in 0..c.epochs {
        let g = crate::solutions::ch02::gradient(model, train)?.gradient;
        for (p, g) in model.iter_mut().zip(g) {
            *p -= c.rate * g;
        }
        curve.push((loss(model, train)?, loss(model, val)?));
    }
    Ok(Run {
        model,
        curve,
        steps: c.epochs,
    })
}
// Supplied deterministic Fisher-Yates permutation; no extra dependency or hidden seed.
pub fn shuffle<T>(rows: &mut [T], state: &mut u64) {
    for i in (1..rows.len()).rev() {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        rows.swap(i, (*state >> 32) as usize % (i + 1));
    }
}
pub type Trainer = fn(&[(f64, f64)], &[(f64, f64)], Config) -> Result<Run, String>;
pub fn report(fit: Trainer, args: &[String]) -> Result<(), String> {
    let base = defaults();
    if !args.is_empty() {
        let o = crate::args::Options::parse(
            args,
            &[
                "--rate",
                "--epochs",
                "--batch",
                "--l2",
                "--seed",
                "--variant",
                "--rows",
                "--patience",
            ],
        )?;
        let c = Config {
            rate: o.number("--rate", base.rate)?,
            epochs: o.count("--epochs", base.epochs)?,
            batch: o.count("--batch", base.batch)?,
            l2: o.number("--l2", base.l2)?,
            seed: o.count("--seed", 7)? as u64,
            patience: o.count("--patience", 0)?,
        };
        let n = o.count("--rows", TRAIN.len())?;
        if n == 0 || n > TRAIN.len() {
            return Err("rows must be1..12".into());
        }
        let mut train = TRAIN[..n].to_vec();
        let mut val = VALIDATION.to_vec();
        match o.text("--variant", "original") {
            "original" => {}
            "rotated" => {
                let mut targets: Vec<_> = train.iter().map(|(_, y)| *y).collect();
                targets.rotate_left(1);
                for ((_, y), new) in train.iter_mut().zip(targets) {
                    *y = new;
                }
            }
            "shifted" => {
                for (x, y) in &mut val {
                    *y = 2. * x.min(2.) + 1.;
                }
            }
            _ => return Err("variant must be original, rotated, or shifted".into()),
        }
        let r = fit(&train, &val, c)?;
        println!("custom configuration={c:?}; rows={n}");
        for (epoch, (train, val)) in r.curve.iter().enumerate() {
            println!("epoch={epoch}, train MSE={train:.6}, validation MSE={val:.6}");
        }
        println!(
            "steps={}, returned model={:?}, returned-model validation MSE={:.6}",
            r.steps,
            r.model,
            loss(r.model, &val)?
        );
        return Ok(());
    }
    for (name, c) in [
        (
            "slow",
            Config {
                rate: 0.001,
                ..base
            },
        ),
        ("full batch", base),
        ("minibatch", Config { batch: 5, ..base }),
        (
            "L2",
            Config {
                batch: 5,
                l2: 0.2,
                ..base
            },
        ),
        (
            "unstable",
            Config {
                rate: 1.,
                epochs: 15,
                ..base
            },
        ),
    ] {
        println!("{name}: {c:?}");
        match fit(&TRAIN, &VALIDATION, c) {
            Ok(r) => {
                for epoch in [0, 1, 5, r.curve.len() - 1] {
                    if let Some((train, val)) = r.curve.get(epoch) {
                        println!("epoch={epoch}, train MSE={train:.6}, validation MSE={val:.6}");
                    }
                }
                println!("steps={}, model={:?}", r.steps, r.model);
            }
            Err(e) => println!("unsupported/failed run: {e}"),
        }
    }
    Ok(())
}
pub fn verify(fit: Trainer) -> Result<(), String> {
    let base = Config {
        batch: 5,
        ..defaults()
    };
    let run = fit(&TRAIN, &VALIDATION, base)?;
    if run.steps != 240 {
        return Err(format!(
            "GOAL_NOT_MET: 80 epochs with batches5,5,2 need240 updates, got{}",
            run.steps
        ));
    }
    let end = loss(run.model, &VALIDATION)?;
    println!("minibatch validation MSE={end:.6}");
    if end > 0.2 {
        return Err("GOAL_NOT_MET: minibatch validation MSE must be below0.2 on fixture".into());
    }
    let reg = fit(&TRAIN, &VALIDATION, Config { l2: 0.2, ..base })?;
    if reg.model[0].abs() >= run.model[0].abs() {
        return Err("GOAL_NOT_MET: L2 should shrink this fitted weight".into());
    }
    // A second update starts at a nonzero weight, so the penalty must contribute.
    let penalty_rows = [(1., 1.)];
    let penalty_config = Config {
        epochs: 1,
        batch: 1,
        rate: 0.1,
        l2: 0.5,
        ..base
    };
    let before = fit(&penalty_rows, &penalty_rows, penalty_config)?.model;
    let after = fit(
        &penalty_rows,
        &penalty_rows,
        Config {
            epochs: 2,
            ..penalty_config
        },
    )?
    .model;
    let objective = |m: Model| loss(m, &penalty_rows).map(|v| v + penalty_config.l2 * m[0] * m[0]);
    for coordinate in 0..2 {
        let mut plus = before;
        let mut minus = before;
        plus[coordinate] += 1e-5;
        minus[coordinate] -= 1e-5;
        let numeric = (objective(plus)? - objective(minus)?) / 2e-5;
        let observed = (before[coordinate] - after[coordinate]) / penalty_config.rate;
        if !crate::scalar::close(observed, numeric) {
            return Err(format!(
                "GOAL_NOT_MET: coordinate {coordinate} update disagrees with the finite-difference L2 objective"
            ));
        }
    }
    let short = fit(
        &[(0., 1.), (0., 1.), (0., 1.)],
        &[(0., 1.)],
        Config {
            epochs: 1,
            batch: 2,
            rate: 0.1,
            l2: 0.5,
            ..base
        },
    )?;
    if short.steps != 2 || (short.model[1] - 0.36).abs() > 1e-10 || short.model[0].abs() > 1e-10 {
        return Err(
            "GOAL_NOT_MET: average the short batch by its own length and leave bias unpenalized"
                .into(),
        );
    }
    let early = fit(
        &TRAIN,
        &VALIDATION,
        Config {
            patience: 5,
            ..base
        },
    )?;
    let minimum = early
        .curve
        .iter()
        .map(|(_, v)| *v)
        .fold(f64::INFINITY, f64::min);
    if !crate::scalar::close(loss(early.model, &VALIDATION)?, minimum) {
        return Err("GOAL_NOT_MET: early stopping must return the saved best model".into());
    }
    // Every update helps training and worsens validation: epoch zero stays best.
    let worsening = fit(
        &[(0., 1.)],
        &[(0., -1.)],
        Config {
            epochs: 10,
            batch: 1,
            rate: 0.1,
            patience: 2,
            ..base
        },
    )?;
    if worsening.steps != 2
        || worsening.curve.len() != 3
        || worsening.model != [0.; 2]
        || worsening.curve[1].1 <= worsening.curve[0].1
        || worsening.curve[2].1 <= worsening.curve[1].1
    {
        return Err(
            "GOAL_NOT_MET: stop after two worsening epochs and return the saved epoch-zero model"
                .into(),
        );
    }
    for seed in [3, 29] {
        let r = fit(&TRAIN, &VALIDATION, Config { seed, ..base })?;
        if loss(r.model, &VALIDATION)? > 0.2 {
            return Err("GOAL_NOT_MET: seed variation failed".into());
        }
    }
    Ok(())
}
pub fn run(args: &[String]) -> Result<(), String> {
    report(fit, args)
}
pub fn check() -> Result<(), String> {
    verify(fit)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_and_goal_solutions() -> Result<(), String> {
        let r = fit(&TRAIN, &VALIDATION, defaults())?;
        assert!(loss(r.model, &VALIDATION)? < 0.2);
        assert!(fit(
            &TRAIN,
            &VALIDATION,
            Config {
                epochs: 0,
                batch: 0,
                ..defaults()
            }
        )
        .is_err());
        assert!(fit(
            &TRAIN,
            &VALIDATION,
            Config {
                epochs: 0,
                l2: -1.0,
                ..defaults()
            }
        )
        .is_err());
        crate::solutions::ch06::check()
    }
    #[test]
    fn goal_checks_reject_wrong_penalty_and_ignored_patience() {
        fn wrong_penalty(
            train: &[(f64, f64)],
            val: &[(f64, f64)],
            mut c: Config,
        ) -> Result<Run, String> {
            c.l2 *= 0.5;
            crate::solutions::ch06::fit(train, val, c)
        }
        fn ignores_patience(
            train: &[(f64, f64)],
            val: &[(f64, f64)],
            mut c: Config,
        ) -> Result<Run, String> {
            if c.patience > 0 {
                c.patience = usize::MAX;
            }
            crate::solutions::ch06::fit(train, val, c)
        }
        assert!(verify(wrong_penalty)
            .unwrap_err()
            .contains("finite-difference L2 objective"));
        assert!(verify(ignores_patience)
            .unwrap_err()
            .contains("two worsening epochs"));
    }
    #[test]
    fn shuffle_preserves_rows() {
        let mut rows = [0, 1, 2, 3, 4, 5];
        shuffle(&mut rows, &mut 7);
        rows.sort();
        assert_eq!(rows, [0, 1, 2, 3, 4, 5]);
    }
}
