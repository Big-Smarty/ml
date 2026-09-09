//! Orthogonal iteration uses P*C*v, where P subtracts already-retained directions.
//! Trying coordinate starts avoids an exactly orthogonal starting-vector blind spot.
use crate::ch17::{self, Pca};
use crate::data::{self, Result};
fn orthogonalize(v: &mut [f64], previous: &[Vec<f64>]) {
    for q in previous {
        let overlap = data::dot(v, q);
        for (v, q) in v.iter_mut().zip(q) {
            *v -= overlap * q;
        }
    }
}
fn norm(v: &[f64]) -> f64 {
    v.iter().fold(0_f64, |n, x| n.hypot(*x))
}
fn direction(c: &[Vec<f64>], previous: &[Vec<f64>], steps: usize) -> Result<(Vec<f64>, f64)> {
    let d = c.len();
    let mut best: Option<(Vec<f64>, f64)> = None;
    for axis in 0..d {
        let mut q = vec![0.; d];
        q[axis] = 1.;
        orthogonalize(&mut q, previous);
        let n = norm(&q);
        // This dimensionless test detects an exhausted coordinate start, not a small eigenvalue.
        if n < 1e-10 {
            continue;
        }
        for v in &mut q {
            *v /= n;
        }
        for _ in 0..steps {
            let mut next = c.iter().map(|r| data::dot(r, &q)).collect::<Vec<_>>();
            orthogonalize(&mut next, previous);
            let n = data::finite(norm(&next))?;
            if n == 0. {
                break;
            }
            for v in &mut next {
                *v /= n;
            }
            let change = data::distance(&q, &next)?;
            q = next;
            if change < 1e-24 {
                break;
            }
        }
        let cq: Vec<_> = c.iter().map(|r| data::dot(r, &q)).collect();
        let value = data::finite(data::dot(&q, &cq))?.max(0.);
        if best.as_ref().is_none_or(|b| value > b.1) {
            best = Some((q, value));
        }
    }
    best.ok_or_else(|| "no independent direction remains; inspect numerical rank".into())
}
pub fn fit(rows: &[Vec<f64>], k: usize, steps: usize) -> Result<Pca> {
    let d = ch17::settings(rows, k, steps)?;
    let mean: Vec<_> = (0..d)
        .map(|j| data::finite(rows.iter().map(|r| r[j] / rows.len() as f64).sum()))
        .collect::<Result<_>>()?;
    let mut covariance = vec![vec![0.; d]; d];
    for (j, row) in covariance.iter_mut().enumerate() {
        for (l, c) in row.iter_mut().enumerate() {
            *c = data::finite(
                rows.iter()
                    .map(|r| (r[j] - mean[j]) * (r[l] - mean[l]) / (rows.len() - 1) as f64)
                    .sum(),
            )?;
        }
    }
    let mut components = Vec::new();
    let mut values = Vec::new();
    // ponytail: explicit covariance and fixed iteration cap suit tiny sensor matrices; use SVD for ill-conditioned large data.
    for _ in 0..k {
        let (q, value) = direction(&covariance, &components, steps)?;
        components.push(q);
        values.push(value);
    }
    Ok(Pca {
        mean,
        covariance,
        components,
        values,
    })
}
pub fn run(_: &[String]) -> Result<()> {
    ch17::report(fit)
}
pub fn check() -> Result<()> {
    ch17::verify(fit)
}
