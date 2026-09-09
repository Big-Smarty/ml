//! Learner: alternate assignment/mean updates, then implement a full EM loop.
//! Supplied density arithmetic keeps component probabilities separate from labels.
use crate::data::{self, Result};
#[derive(Debug)]
pub struct KMeans {
    pub centers: Vec<Vec<f64>>,
    pub assignments: Vec<usize>,
    pub inertia: f64,
}
#[derive(Clone, Debug)]
pub struct Component {
    pub weight: f64,
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
}
#[derive(Debug)]
pub struct Gmm {
    pub components: Vec<Component>,
    pub likelihood: Vec<f64>,
}
pub type KFit = fn(&[Vec<f64>], usize, usize) -> Result<KMeans>;
pub type GFit = fn(&[Vec<f64>], usize, usize, f64) -> Result<Gmm>;
pub fn settings(rows: &[Vec<f64>], k: usize, steps: usize) -> Result<usize> {
    let d = data::validate_vectors(rows, 2)?;
    if k == 0 || k > rows.len() || k > 16 || steps == 0 || steps > 1000 {
        return Err("need1..min(n,16) components and1..1000 steps".into());
    }
    Ok(d)
}
pub fn assign(rows: &[Vec<f64>], centers: &[Vec<f64>]) -> Result<(Vec<usize>, f64)> {
    if centers.is_empty() {
        return Err("need centers".into());
    }
    let mut assignments = Vec::new();
    let mut inertia = 0.;
    for x in rows {
        let mut best = (f64::INFINITY, 0);
        for (j, c) in centers.iter().enumerate() {
            let d = data::distance(x, c)?;
            if d < best.0 {
                best = (d, j);
            }
        }
        assignments.push(best.1);
        inertia += best.0;
    }
    Ok((assignments, data::finite(inertia)?))
}
pub fn kmeans(rows: &[Vec<f64>], k: usize, steps: usize) -> Result<KMeans> {
    settings(rows, k, steps)?;
    // Working fixed prototypes: assignment is useful, but centers do not learn yet.
    let centers: Vec<_> = (0..k)
        .map(|j| rows[j * (rows.len() - 1) / (k - 1).max(1)].clone())
        .collect();
    let (assignments, inertia) = assign(rows, &centers)?;
    Ok(KMeans {
        centers,
        assignments,
        inertia,
    })
}
pub fn initial(centers: Vec<Vec<f64>>) -> Gmm {
    let k = centers.len();
    Gmm {
        components: centers
            .into_iter()
            .map(|mean| Component {
                weight: 1. / k as f64,
                variance: vec![1.; mean.len()],
                mean,
            })
            .collect(),
        likelihood: vec![],
    }
}
impl Gmm {
    pub fn log_terms(&self, x: &[f64]) -> Result<Vec<f64>> {
        if self.components.is_empty() {
            return Err("mixture needs components".into());
        }
        data::query(x, self.components[0].mean.len())?;
        self.components
            .iter()
            .map(|c| {
                let density = x
                    .iter()
                    .enumerate()
                    .map(|(j, x)| {
                        -0.5 * ((2. * std::f64::consts::PI).ln()
                            + c.variance[j].ln()
                            + (x - c.mean[j]).powi(2) / c.variance[j])
                    })
                    .sum::<f64>();
                data::finite(density)?;
                Ok(c.weight.ln() + density)
            })
            .collect()
    }
    pub fn responsibilities(&self, x: &[f64]) -> Result<Vec<f64>> {
        let terms = self.log_terms(x)?;
        let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let masses: Vec<_> = terms.iter().map(|v| (v - max).exp()).collect();
        let sum = data::finite(masses.iter().sum())?;
        if sum <= 0. {
            return Err("mixture has no mass".into());
        }
        Ok(masses.iter().map(|p| p / sum).collect())
    }
    pub fn log_density(&self, x: &[f64]) -> Result<f64> {
        let terms = self.log_terms(x)?;
        let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        data::finite(max + terms.iter().map(|v| (v - max).exp()).sum::<f64>().ln())
    }
    pub fn log_likelihood(&self, rows: &[Vec<f64>]) -> Result<f64> {
        let mut sum = 0.;
        for x in rows {
            sum += self.log_density(x)?;
        }
        data::finite(sum)
    }
}
pub fn gmm(rows: &[Vec<f64>], k: usize, steps: usize, floor: f64) -> Result<Gmm> {
    settings(rows, k, steps)?;
    if !floor.is_finite() || floor <= 0. {
        return Err("variance floor must be positive and finite".into());
    }
    // Working unit-variance mixture, with fixed prototype centers and equal weights.
    let mut model = initial(kmeans(rows, k, steps)?.centers);
    model.likelihood.push(model.log_likelihood(rows)?);
    Ok(model)
}
pub fn run(_: &[String]) -> Result<()> {
    report(kmeans, gmm)
}
pub fn check() -> Result<()> {
    verify(kmeans, gmm)
}
pub fn report(kmeans: KFit, gmm: GFit) -> Result<()> {
    let (train, valid) = crate::ch17::sensor_space()?;
    let pca = crate::solutions::ch17::fit(&train, 2, 400)?;
    let train = train
        .iter()
        .map(|x| pca.transform(x))
        .collect::<Result<Vec<_>>>()?;
    let valid = valid
        .iter()
        .map(|x| pca.transform(x))
        .collect::<Result<Vec<_>>>()?;
    for k in [2, 3] {
        let m = kmeans(&train, k, 30)?;
        println!(
            "k-means k{k} inertia(sum)={:.4} centers={:?} counts={:?}",
            m.inertia,
            m.centers,
            (0..k)
                .map(|j| m.assignments.iter().filter(|&&a| a == j).count())
                .collect::<Vec<_>>()
        );
    }
    let model = gmm(&train, 2, 30, 0.02)?;
    println!(
        "EM summed log likelihood, first={:.4} last={:.4}; weights={:?}",
        model.likelihood[0],
        model.likelihood[model.likelihood.len() - 1],
        model
            .components
            .iter()
            .map(|c| c.weight)
            .collect::<Vec<_>>()
    );
    let mut scores = valid
        .iter()
        .enumerate()
        .map(|(i, x)| model.log_density(x).map(|d| (i, -d)))
        .collect::<Result<Vec<_>>>()?;
    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!(
        "highest development anomaly scores (validation index, score): {:?}",
        &scores[..3]
    );
    println!("First validation responsibilities={:?}. These operating-regime densities are not failure probabilities.",model.responsibilities(&valid[0])?);
    Ok(())
}
pub fn verify(kmeans: KFit, gmm: GFit) -> Result<()> {
    let rows = vec![vec![1., 0.], vec![2., 0.], vec![8., 0.], vec![9., 0.]];
    let k = kmeans(&rows, 2, 10)?;
    if (k.inertia - 1.).abs() > 1e-10 {
        return data::goal(format!(
            "Goal not met: centers1.5,8.5 yield summed inertia1; got{}",
            k.inertia
        ));
    }
    let tricky = vec![vec![0.], vec![4.], vec![5.], vec![5.1], vec![10.]];
    let k = kmeans(&tricky, 2, 1)?;
    if k.assignments[3] != 0 {
        return data::goal("recompute assignments after the final center update".into());
    }
    let rows: Vec<_> = [-3., -2.5, -2., -1.5, -1., 1.9, 2., 2.1]
        .into_iter()
        .map(|x| vec![x, 0.])
        .collect();
    let m = gmm(&rows, 2, 50, 0.01)?;
    if m.likelihood.len() < 2 || m.likelihood[m.likelihood.len() - 1] <= m.likelihood[0] + 0.1 {
        return data::goal("Goal not met: a full EM fit must update mixture parameters and record likelihood progress".into());
    }
    if m.likelihood.windows(2).any(|p| p[1] + 1e-8 < p[0]) {
        return data::goal("EM likelihood decreased beyond1e-8 on the well-conditioned check; inspect old responsibilities/new means".into());
    }
    let mass = m.components.iter().map(|c| c.weight).sum::<f64>();
    if (mass - 1.).abs() > 1e-12
        || m.components
            .iter()
            .any(|c| c.variance.iter().any(|&v| v < 0.01))
    {
        return data::goal("mixture weights normalize and variances honor their floor".into());
    }
    if -m.log_density(&[15., 0.])? <= -m.log_density(&[2., 0.])? {
        return data::goal("far unfamiliar input should have greater negative log density".into());
    }
    let shared = Component {
        weight: 0.5,
        mean: vec![0., 0.],
        variance: vec![1., 1.],
    };
    let m = Gmm {
        components: vec![shared.clone(), shared],
        likelihood: vec![],
    };
    if m.responsibilities(&[1e8, 1e8])? != [0.5, 0.5] {
        return data::goal("normalize shifted masses directly to preserve equal responsibilities at huge common offsets".into());
    }
    println!("Center updates, final assignments, unequal-spread EM, likelihood, floors, and unfamiliar anomaly checks passed.");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_and_solution() -> Result<()> {
        let (t, _) = crate::ch17::sensor_space()?;
        assert!(gmm(&t, 2, 3, 0.01)?.log_density(&t[0])?.is_finite());
        assert!(gmm(&t, 2, 3, 0.).is_err());
        crate::solutions::ch18::check()
    }
}
