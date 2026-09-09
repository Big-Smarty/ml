//! Each batch averages by its ACTUAL length. L2 modifies the objective once per step.
//! Save a model together with its validation loss; return that model when stopping.
use crate::{
    ch06::{self, Config, Run},
    scalar::loss,
};
pub fn fit(train: &[(f64, f64)], val: &[(f64, f64)], c: Config) -> Result<Run, String> {
    ch06::validate(train, val, c)?;
    let mut model = [0.; 2];
    let mut steps = 0;
    let mut curve = vec![(loss(model, train)?, loss(model, val)?)];
    let mut rows = train.to_vec();
    let mut state = c.seed;
    let mut best = model;
    let mut best_loss = loss(model, val)?;
    let mut stale = 0;
    for _ in 0..c.epochs {
        ch06::shuffle(&mut rows, &mut state);
        for batch in rows.chunks(c.batch) {
            let mut g = crate::solutions::ch02::gradient(model, batch)?.gradient;
            g[0] += 2. * c.l2 * model[0];
            steps += 1;
            for (parameter, slope) in model.iter_mut().zip(g) {
                *parameter -= c.rate * slope;
            }
            loss(model, batch)?;
        }
        curve.push((loss(model, train)?, loss(model, val)?));
        let validation = loss(model, val)?;
        if validation < best_loss {
            best_loss = validation;
            best = model;
            stale = 0;
        } else {
            stale += 1;
        }
        if c.patience > 0 && stale >= c.patience {
            break;
        }
    }
    Ok(Run {
        model: if c.patience > 0 { best } else { model },
        curve,
        steps,
    })
}
pub fn run(args: &[String]) -> Result<(), String> {
    ch06::report(fit, args)
}
pub fn check() -> Result<(), String> {
    ch06::verify(fit)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn l2_gradient_and_short_batch() -> Result<(), String> {
        let c = Config {
            epochs: 1,
            batch: 2,
            rate: 0.1,
            l2: 0.5,
            ..ch06::defaults()
        };
        let data = [(0., 1.), (0., 1.), (0., 1.)];
        let run = fit(&data, &data, c)?;
        assert_eq!(run.steps, 2);
        assert!((run.model[1] - 0.36).abs() < 1e-12);
        assert_eq!(run.model[0], 0.);
        let m = [-2., 1.];
        let sample = [(1., 0.)];
        let analytic = crate::solutions::ch02::gradient(m, &sample)?.gradient[0] + 2. * c.l2 * m[0];
        let objective = |w| loss([w, m[1]], &sample).map(|v| v + c.l2 * w * w);
        let numeric = (objective(m[0] + 1e-5)? - objective(m[0] - 1e-5)?) / 2e-5;
        assert!(crate::scalar::close(analytic, numeric));
        Ok(())
    }
}
