//! Learner: change perceptron mistakes into margin-violation SVM updates, then
//! learn all kernel coefficients instead of using two fixed class prototypes.
use crate::data::{self, Point, Result};
#[derive(Debug)]
pub struct Linear {
    pub weights: Vec<f64>,
    pub bias: f64,
}
impl Linear {
    pub fn score(&self, x: &[f64]) -> Result<f64> {
        data::query(x, self.weights.len())?;
        data::finite(data::dot(&self.weights, x) + self.bias)
    }
    pub fn objective(&self, rows: &[Point], lambda: f64) -> Result<f64> {
        data::validate_points(rows)?;
        if !lambda.is_finite() || lambda < 0. {
            return Err("invalid regularization".into());
        }
        let hinge = rows
            .iter()
            .map(|r| {
                self.score(&r.features)
                    .map(|s| (1. - signed(r.label) * s).max(0.))
            })
            .collect::<Result<Vec<_>>>()?;
        data::finite(
            hinge.iter().sum::<f64>() / rows.len() as f64
                + 0.5 * lambda * data::dot(&self.weights, &self.weights),
        )
    }
}
pub fn signed(label: u8) -> f64 {
    2. * f64::from(label) - 1.
}
pub type SvmFit = fn(&[Point], usize, f64, f64) -> Result<Linear>;
pub type KernelFit = fn(&[Point], usize, f64) -> Result<Kernel>;
#[derive(Debug)]
pub struct Kernel {
    pub rows: Vec<Point>,
    pub alpha: Vec<f64>,
    pub gamma: f64,
}
impl Kernel {
    pub fn score(&self, x: &[f64]) -> Result<f64> {
        let mut score = 0.;
        for (r, &a) in self.rows.iter().zip(&self.alpha) {
            score += a * signed(r.label) * rbf(&r.features, x, self.gamma)?;
        }
        data::finite(score)
    }
}
pub fn rbf(a: &[f64], b: &[f64], gamma: f64) -> Result<f64> {
    if !gamma.is_finite() || gamma <= 0. {
        return Err("gamma must be finite and positive".into());
    }
    Ok((-gamma * data::distance(a, b)?).exp())
}
pub fn settings(epochs: usize, rate: f64, lambda: f64) -> Result<()> {
    if epochs == 0
        || epochs > 10_000
        || !rate.is_finite()
        || rate <= 0.
        || !lambda.is_finite()
        || lambda < 0.
    {
        Err("invalid bounded training settings".into())
    } else {
        Ok(())
    }
}
pub fn svm(rows: &[Point], epochs: usize, rate: f64, lambda: f64) -> Result<Linear> {
    let d = data::validate_points(rows)?;
    settings(epochs, rate, lambda)?;
    let mut model = Linear {
        weights: vec![0.; d],
        bias: 0.,
    };
    for _ in 0..epochs {
        for row in rows {
            let y = signed(row.label);
            // Working perceptron baseline: correct signs stop updating, however narrow the margin.
            if y * model.score(&row.features)? <= 0. {
                for (w, x) in model.weights.iter_mut().zip(&row.features) {
                    *w += rate * y * x;
                }
                model.bias += rate * y;
            }
        }
    }
    model.objective(rows, lambda)?;
    Ok(model)
}
pub fn kernel(rows: &[Point], epochs: usize, gamma: f64) -> Result<Kernel> {
    data::validate_points(rows)?;
    settings(epochs, 0.1, 0.)?;
    rbf(&rows[0].features, &rows[0].features, gamma)?;
    let mut chosen = Vec::new();
    for label in [0, 1] {
        if let Some(r) = rows.iter().find(|r| r.label == label) {
            chosen.push(r.clone());
        }
    }
    // Working two-prototype RBF rule, without learned correction coefficients.
    Ok(Kernel {
        alpha: vec![1.; chosen.len()],
        rows: chosen,
        gamma,
    })
}
pub fn run(_: &[String]) -> Result<()> {
    report(svm, kernel)
}
pub fn check() -> Result<()> {
    verify(svm, kernel)
}
pub fn report(svm: SvmFit, kernel: KernelFit) -> Result<()> {
    let (raw, train, valid) = crate::ch14::prepared()?;
    for lambda in [0.01, 0.2] {
        let m = svm(&train, 120, 0.02, lambda)?;
        let p = valid
            .iter()
            .map(|r| m.score(&r.features).map(|s| f64::from(s >= 0.)))
            .collect::<Result<Vec<_>>>()?;
        crate::evaluation::show(
            &format!("linear decision lambda{lambda}"),
            &data::predictions(&raw, &p)?,
        )?;
        println!(
            " training hinge+penalty {:.4}; norm {:.4}",
            m.objective(&train, lambda)?,
            data::dot(&m.weights, &m.weights).sqrt()
        );
    }
    for gamma in [0.05, 0.5, 5.] {
        let m = kernel(&train, 12, gamma)?;
        let p = valid
            .iter()
            .map(|r| m.score(&r.features).map(|s| f64::from(s >= 0.)))
            .collect::<Result<Vec<_>>>()?;
        crate::evaluation::show(
            &format!("RBF decision gamma{gamma}"),
            &data::predictions(&raw, &p)?,
        )?;
        println!(
            " nonzero support coefficients {}",
            m.alpha.iter().filter(|&&a| a > 0.).count()
        );
    }
    println!("Brier above uses hard0/1 decisions, not calibrated SVM probabilities. Nonlinear training goal is a kernel perceptron, not a kernel SVM dual solver.");
    Ok(())
}
pub fn verify(svm: SvmFit, kernel: KernelFit) -> Result<()> {
    let r = [Point {
        features: vec![1., 0.],
        label: 1,
    }];
    let m = svm(&r, 2, 0.25, 0.1)?;
    if (m.weights[0] - 0.49375).abs() > 1e-12 || (m.bias - 0.5).abs() > 1e-12 {
        return data::goal(format!("Goal not met: positive-but-small margin still updates; expected w0=.49375,b=.5, got {m:?}"));
    }
    let m = svm(&r, 3, 0.5, 0.1)?;
    if (m.weights[0] - 0.95125).abs() > 1e-12 || (m.bias - 1.).abs() > 1e-12 {
        return data::goal("shrink weights on every step and test the margin before shrinking; leave bias unpenalized".into());
    }
    let rows: Vec<_> = [(-1., -1., 1), (-1., 1., 0), (1., -1., 0), (1., 1., 1)]
        .into_iter()
        .map(|(a, b, label)| Point {
            features: vec![a, b],
            label,
        })
        .collect();
    let k = kernel(&rows, 12, 1.)?;
    for r in &rows {
        let query: Vec<_> = r.features.iter().map(|x| x * 0.85).collect();
        if signed(r.label) * k.score(&query)? <= 0. {
            return data::goal("Goal not met: learned kernel coefficients must generalize locally to all four XOR corners".into());
        }
    }
    if (rbf(&[0., 0.], &[2., 0.], 0.5)? - (-2_f64).exp()).abs() > 1e-12 {
        return data::goal("RBF uses squared distance".into());
    }
    println!("Active/inactive hinge updates, weight-only shrinkage, and perturbed nonlinear queries passed.");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_and_solution() -> Result<()> {
        let (_, t, _) = crate::ch14::prepared()?;
        assert!(svm(&t, 2, 0.01, 0.1)?.objective(&t, 0.1)?.is_finite());
        assert!(svm(&t, 1, f64::NAN, 0.1).is_err());
        crate::solutions::ch16::check()
    }
}
