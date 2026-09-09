//! Learner: expand nearest-one voting to k neighbors, and centroid fitting to
//! class priors and per-feature Gaussian variances. Shared preprocessing is a checkpoint.
use crate::data::{self, Point, Result};
pub type Neighbors = fn(&[Point], &[f64], usize) -> Result<f64>;
pub type Bayes = fn(&[Point], f64) -> Result<GaussianNb>;
#[derive(Debug)]
pub struct GaussianNb {
    pub prior: [f64; 2],
    pub mean: [Vec<f64>; 2],
    pub variance: [Vec<f64>; 2],
}
impl GaussianNb {
    pub fn probability(&self, x: &[f64]) -> Result<f64> {
        data::query(x, self.mean[0].len())?;
        let mut scores = [0.; 2];
        for (c, s) in scores.iter_mut().enumerate() {
            *s = self.prior[c].ln();
            for (j, value) in x.iter().enumerate() {
                *s -= 0.5
                    * ((2. * std::f64::consts::PI).ln()
                        + self.variance[c][j].ln()
                        + (value - self.mean[c][j]).powi(2) / self.variance[c][j]);
            }
            data::finite(*s)?;
        }
        let max = scores[0].max(scores[1]);
        let a = (scores[0] - max).exp();
        let b = (scores[1] - max).exp();
        Ok(b / (a + b))
    }
}
pub fn knn(rows: &[Point], query: &[f64], k: usize) -> Result<f64> {
    let d = data::validate_points(rows)?;
    data::query(query, d)?;
    if k == 0 || k > rows.len() {
        return Err("k must be in 1..=training count".into());
    }
    // Working nearest-one baseline. Replace the entire selection/vote, not just k.
    let mut nearest = (f64::INFINITY, 0);
    for row in rows {
        let distance = data::distance(&row.features, query)?;
        if distance < nearest.0 {
            nearest = (distance, row.label);
        }
    }
    Ok(f64::from(nearest.1))
}
pub fn bayes(rows: &[Point], floor: f64) -> Result<GaussianNb> {
    initial_bayes(rows, floor)
}

pub fn initial_bayes(rows: &[Point], floor: f64) -> Result<GaussianNb> {
    let d = data::validate_points(rows)?;
    if !floor.is_finite() || floor <= 0. {
        return Err("variance floor must be positive and finite".into());
    }
    let mut count = [0; 2];
    let mut mean = [vec![0.; d], vec![0.; d]];
    for r in rows {
        let c = r.label as usize;
        count[c] += 1;
        for (m, x) in mean[c].iter_mut().zip(&r.features) {
            *m += x;
        }
    }
    if count.contains(&0) {
        return Err("each class needs observed training rows".into());
    }
    for c in 0..2 {
        for m in &mut mean[c] {
            *m = data::finite(*m / count[c] as f64)?;
        }
    }
    // Equal priors, unit spread: useful nearest-centroid Gaussian baseline.
    Ok(GaussianNb {
        prior: [0.5; 2],
        mean,
        variance: [vec![1.; d], vec![1.; d]],
    })
}
pub fn run(_: &[String]) -> Result<()> {
    report(knn, bayes)
}
pub fn check() -> Result<()> {
    verify(knn, bayes)
}
pub fn prepared() -> Result<(Vec<crate::data::RawRow>, Vec<Point>, Vec<Point>)> {
    let rows = data::development(&data::load()?);
    let (train, valid) = crate::solutions::ch13::split(&rows, crate::ch13::Split::FutureGroup)?;
    let prep = crate::solutions::ch13::fit(&train)?;
    Ok((
        valid.clone(),
        prep.transform_all(&train)?,
        prep.transform_all(&valid)?,
    ))
}
pub fn report(knn: Neighbors, bayes: Bayes) -> Result<()> {
    let (raw, train, valid) = prepared()?;
    println!("Maintenance checkpoint: {} training rows, {} features, {} future unseen-machine validation rows",train.len(),train[0].features.len(),valid.len());
    for k in [1, 3, 7] {
        let p = valid
            .iter()
            .map(|r| knn(&train, &r.features, k))
            .collect::<Result<Vec<_>>>()?;
        crate::evaluation::show(&format!("kNN k={k}"), &data::predictions(&raw, &p)?)?;
    }
    let model = bayes(&train, 0.05)?;
    let p = valid
        .iter()
        .map(|r| model.probability(&r.features))
        .collect::<Result<Vec<_>>>()?;
    crate::evaluation::show("Gaussian model", &data::predictions(&raw, &p)?)?;
    println!(
        "NB priors {:?}; first sensor class means [{:.3},{:.3}], variances [{:.3},{:.3}]",
        model.prior, model.mean[0][0], model.mean[1][0], model.variance[0][0], model.variance[1][0]
    );
    println!("kNN scans {} stored rows per query; NB scores two class summaries. Neither normalized output is guaranteed calibrated.",train.len());
    Ok(())
}
pub fn verify(knn: Neighbors, bayes: Bayes) -> Result<()> {
    let rows = vec![
        Point {
            features: vec![0., 0.],
            label: 1,
        },
        Point {
            features: vec![0.2, 0.],
            label: 0,
        },
        Point {
            features: vec![0.3, 0.],
            label: 0,
        },
    ];
    let p = knn(&rows, &[0.01, 0.], 3)?;
    if (p - 1. / 3.).abs() > 1e-12 {
        return data::goal(format!(
            "Goal not met: three-neighbor vote must be 1/3 despite nearest positive; got {p}"
        ));
    }
    let rows: Vec<_> = [(0., 0), (2., 0), (4., 0), (10., 1), (12., 1)]
        .into_iter()
        .map(|(x, label)| Point {
            features: vec![x, 7.],
            label,
        })
        .collect();
    let model = bayes(&rows, 0.01)?;
    if (model.prior[0] - 0.6).abs() > 1e-12
        || (model.variance[0][0] - 8. / 3.).abs() > 1e-12
        || model.variance[1][1] != 0.01
    {
        return data::goal("Goal not met: learn class frequencies and population variances, then max with the supplied floor".into());
    }
    if model.probability(&[11., 7.])? < 0.99 || model.probability(&[1., 7.])? > 0.01 {
        return data::goal(
            "unfamiliar Gaussian class queries should favor their local class".into(),
        );
    }
    if knn(&rows, &[0.], 1).is_ok() || knn(&rows, &[0., 0.], 0).is_ok() || bayes(&[], 0.01).is_ok()
    {
        return data::goal("retain the shape/settings boundaries".into());
    }
    println!("k-dependent voting, learned priors, unequal class spread, constant variance floor and new queries passed.");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_and_solution() -> Result<()> {
        report(knn, bayes)?;
        crate::solutions::ch14::check()
    }
}
