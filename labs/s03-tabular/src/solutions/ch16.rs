//! The hinge condition uses old parameters. Shrink every weight every step;
//! the data term acts on margin violations and the bias has no norm penalty.
use crate::ch16::{self, signed, Kernel, Linear};
use crate::data::{self, Point, Result};
pub fn svm(rows: &[Point], epochs: usize, rate: f64, lambda: f64) -> Result<Linear> {
    let d = data::validate_points(rows)?;
    ch16::settings(epochs, rate, lambda)?;
    let mut model = Linear {
        weights: vec![0.; d],
        bias: 0.,
    };
    for _ in 0..epochs {
        for row in rows {
            let y = signed(row.label);
            let active = y * model.score(&row.features)? < 1.;
            for (w, x) in model.weights.iter_mut().zip(&row.features) {
                *w = data::finite(
                    (1. - rate * lambda) * *w + if active { rate * y * x } else { 0. },
                )?;
            }
            if active {
                model.bias = data::finite(model.bias + rate * y)?;
            }
        }
    }
    model.objective(rows, lambda)?;
    Ok(model)
}
pub fn kernel(rows: &[Point], epochs: usize, gamma: f64) -> Result<Kernel> {
    data::validate_points(rows)?;
    ch16::settings(epochs, 0.1, 0.)?;
    ch16::rbf(&rows[0].features, &rows[0].features, gamma)?;
    let mut model = Kernel {
        rows: rows.to_vec(),
        alpha: vec![0.; rows.len()],
        gamma,
    };
    // ponytail: on-demand O(epochs*n^2*d) similarities suit 144 rows; cache a Gram matrix if repeated evaluation dominates.
    for _ in 0..epochs {
        for (i, row) in rows.iter().enumerate() {
            if signed(row.label) * model.score(&row.features)? <= 0. {
                model.alpha[i] += 1.;
            }
        }
    }
    Ok(model)
}
pub fn run(_: &[String]) -> Result<()> {
    ch16::report(svm, kernel)
}
pub fn check() -> Result<()> {
    ch16::verify(svm, kernel)
}
