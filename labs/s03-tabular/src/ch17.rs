//! Learner: replace coordinate selection with centered covariance, orthogonal
//! power iteration and fitted principal components. Projection plumbing is supplied.
use crate::data::{self, Result};
#[derive(Debug)]
pub struct Pca {
    pub mean: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub components: Vec<Vec<f64>>,
    pub values: Vec<f64>,
}
pub type Fit = fn(&[Vec<f64>], usize, usize) -> Result<Pca>;
impl Pca {
    pub fn transform(&self, x: &[f64]) -> Result<Vec<f64>> {
        data::query(x, self.mean.len())?;
        let centered: Vec<_> = x.iter().zip(&self.mean).map(|(x, m)| x - m).collect();
        self.components
            .iter()
            .map(|q| data::finite(data::dot(&centered, q)))
            .collect()
    }
    pub fn reconstruct(&self, scores: &[f64]) -> Result<Vec<f64>> {
        data::query(scores, self.components.len())?;
        let mut x = self.mean.clone();
        for (score, q) in scores.iter().zip(&self.components) {
            for (x, q) in x.iter_mut().zip(q) {
                *x = data::finite(*x + score * q)?;
            }
        }
        Ok(x)
    }
    pub fn error(&self, rows: &[Vec<f64>]) -> Result<f64> {
        data::validate_vectors(rows, 1)?;
        let mut total = 0.;
        for x in rows {
            total += data::distance(x, &self.reconstruct(&self.transform(x)?)?)?;
        }
        data::finite(total / rows.len() as f64)
    }
    pub fn retained(&self) -> f64 {
        let trace: f64 = self.covariance.iter().enumerate().map(|(j, r)| r[j]).sum();
        if trace == 0. {
            0.
        } else {
            self.values.iter().sum::<f64>() / trace
        }
    }
}
pub fn settings(rows: &[Vec<f64>], k: usize, steps: usize) -> Result<usize> {
    let d = data::validate_vectors(rows, 2)?;
    if k == 0 || k > d || d > 32 || steps == 0 || steps > 10_000 {
        return Err("PCA needs1..d components,d<=32,and1..10000 iterations".into());
    }
    Ok(d)
}
pub fn fit(rows: &[Vec<f64>], k: usize, steps: usize) -> Result<Pca> {
    let d = settings(rows, k, steps)?;
    let mean: Vec<_> = (0..d)
        .map(|j| data::finite(rows.iter().map(|r| r[j] / rows.len() as f64).sum()))
        .collect::<Result<_>>()?;
    let mut covariance = vec![vec![0.; d]; d];
    for (j, row) in covariance.iter_mut().enumerate() {
        row[j] = data::finite(
            rows.iter()
                .map(|r| (r[j] - mean[j]).powi(2) / (rows.len() - 1) as f64)
                .sum(),
        )?;
    }
    let mut axes: Vec<_> = (0..d).collect();
    axes.sort_by(|&a, &b| covariance[b][b].total_cmp(&covariance[a][a]));
    axes.truncate(k);
    let values = axes.iter().map(|&j| covariance[j][j]).collect();
    let components = axes
        .into_iter()
        .map(|j| {
            let mut q = vec![0.; d];
            q[j] = 1.;
            q
        })
        .collect();
    // Working compressor: retain high-variance original coordinates; ignores correlations.
    Ok(Pca {
        mean,
        covariance,
        components,
        values,
    })
}
pub fn run(_: &[String]) -> Result<()> {
    report(fit)
}
pub fn check() -> Result<()> {
    verify(fit)
}
pub type SensorRows = Vec<Vec<f64>>;
pub fn sensor_space() -> Result<(SensorRows, SensorRows)> {
    let (_, train, valid) = crate::ch14::prepared()?;
    Ok((
        train.iter().map(|r| r.features[..3].to_vec()).collect(),
        valid.iter().map(|r| r.features[..3].to_vec()).collect(),
    ))
}
pub fn report(fit: Fit) -> Result<()> {
    let (train, valid) = sensor_space()?;
    println!("PCA reads the three training-standardized continuous sensors; excludes category/indicator columns and every label.");
    for k in [1, 2, 3] {
        let p = fit(&train, k, 400)?;
        println!("k={k}: eigenvalues={:?}, retained={:.4}, train squared reconstruction norm/row={:.4}, validation={:.4}",p.values,p.retained(),p.error(&train)?,p.error(&valid)?);
        if k == 3 {
            let min = p.values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = p.values.iter().copied().fold(0., f64::max);
            println!(
                "covariance={:?}; condition number={}; projection first validation={:?}",
                p.covariance,
                if min <= 0. { f64::INFINITY } else { max / min },
                p.transform(&valid[0])?
            );
        }
    }
    Ok(())
}
pub fn verify(fit: Fit) -> Result<()> {
    let s = 2_f64.sqrt();
    let rows = vec![vec![s, 0.], vec![0., s], vec![-s, -s]];
    let p = fit(&rows, 2, 400)?;
    if (p.covariance[0][1] - 1.).abs() > 1e-12
        || (p.values[0] - 3.).abs() > 1e-8
        || (p.values[1] - 1.).abs() > 1e-8
    {
        return data::goal(format!(
            "Goal not met: full centered covariance [[2,1],[1,2]] has eigenvalues[3,1]; got {:?}",
            p.values
        ));
    }
    for (q, &lambda) in p.components.iter().zip(&p.values) {
        let cq: Vec<_> = p.covariance.iter().map(|r| data::dot(r, q)).collect();
        let lq: Vec<_> = q.iter().map(|q| lambda * q).collect();
        if (data::dot(q, q) - 1.).abs() > 1e-10 || data::distance(&cq, &lq)? > 1e-14 {
            return data::goal(
                "each component must be unit norm with small eigenpair residual".into(),
            );
        }
    }
    if data::dot(&p.components[0], &p.components[1]).abs() > 1e-10 || p.error(&rows)? > 1e-16 {
        return data::goal("orthogonal components must reconstruct the complete2D space".into());
    }
    let one = fit(&rows, 1, 400)?;
    if (one.retained() - 0.75).abs() > 1e-8 {
        return data::goal("one component retains3/4 variance".into());
    }
    let shifted: Vec<_> = rows.iter().map(|r| vec![r[0] + 100., r[1] - 50.]).collect();
    let q = fit(&shifted, 1, 400)?;
    if (q.error(&shifted)? - one.error(&rows)?).abs() > 1e-10 {
        return data::goal("translation should not change centered reconstruction error".into());
    }
    let rank_one = vec![vec![-1., 2., -3.], vec![0., 0., 0.], vec![1., -2., 3.]];
    let q = fit(&rank_one, 2, 400)?;
    if q.error(&rank_one)? > 1e-16 {
        return data::goal(
            "rank-one3D data must reconstruct with one informative component".into(),
        );
    }
    let zero = fit(&[vec![3., 2.], vec![3., 2.]], 2, 30)?;
    if zero.error(&[vec![3., 2.]])? != 0. {
        return data::goal("constant data has no preferred axis but reconstructs exactly".into());
    }
    println!("Covariance, eigenpair residuals, orthogonality, translation,3D rank deficiency and constant-data checks passed.");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_and_solution() -> Result<()> {
        let (t, v) = sensor_space()?;
        assert!(fit(&t, 2, 20)?.error(&v)?.is_finite());
        assert!(fit(&[], 1, 20).is_err());
        crate::solutions::ch17::check()
    }
}
