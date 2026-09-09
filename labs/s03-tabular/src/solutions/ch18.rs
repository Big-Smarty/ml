//! Lloyd alternates exact assignments and means. EM freezes every E-step
//! responsibility before using fractional counts for all M-step components.
use crate::ch18::{self, Component, Gmm, KMeans};
use crate::data::{self, Result};
pub fn kmeans(rows: &[Vec<f64>], k: usize, steps: usize) -> Result<KMeans> {
    let d = ch18::settings(rows, k, steps)?;
    let mut centers: Vec<_> = (0..k)
        .map(|j| rows[j * (rows.len() - 1) / (k - 1).max(1)].clone())
        .collect();
    for _ in 0..steps {
        let (assignments, _) = ch18::assign(rows, &centers)?;
        let mut sums = vec![vec![0.; d]; k];
        let mut counts = vec![0; k];
        for (x, &j) in rows.iter().zip(&assignments) {
            counts[j] += 1;
            for (s, x) in sums[j].iter_mut().zip(x) {
                *s += x;
            }
        }
        for j in 0..k {
            // ponytail: retain an empty center; farthest-point reseeding if initialization leaves dead clusters.
            if counts[j] > 0 {
                for (c, s) in centers[j].iter_mut().zip(&sums[j]) {
                    *c = data::finite(s / counts[j] as f64)?;
                }
            }
        }
    }
    let (assignments, inertia) = ch18::assign(rows, &centers)?;
    Ok(KMeans {
        centers,
        assignments,
        inertia,
    })
}
pub fn gmm(rows: &[Vec<f64>], k: usize, steps: usize, floor: f64) -> Result<Gmm> {
    let d = ch18::settings(rows, k, steps)?;
    if !floor.is_finite() || floor <= 0. {
        return Err("variance floor must be positive and finite".into());
    }
    let mut model = ch18::initial(kmeans(rows, k, 8)?.centers);
    model.likelihood.push(model.log_likelihood(rows)?);
    for _ in 0..steps {
        let responsibilities = rows
            .iter()
            .map(|x| model.responsibilities(x))
            .collect::<Result<Vec<_>>>()?;
        for j in 0..k {
            let mass: f64 = responsibilities.iter().map(|r| r[j]).sum();
            if mass <= 1e-12 {
                model.components[j].weight = 0.;
                continue;
            }
            let mut mean = vec![0.; d];
            for (x, r) in rows.iter().zip(&responsibilities) {
                for (m, x) in mean.iter_mut().zip(x) {
                    *m += r[j] * x / mass;
                }
            }
            let mut variance = vec![0.; d];
            for (x, r) in rows.iter().zip(&responsibilities) {
                for ((v, x), m) in variance.iter_mut().zip(x).zip(&mean) {
                    *v += r[j] * (x - m).powi(2) / mass;
                }
            }
            for v in &mut variance {
                *v = data::finite(*v)?.max(floor);
            }
            for &m in &mean {
                data::finite(m)?;
            }
            model.components[j] = Component {
                weight: mass / rows.len() as f64,
                mean,
                variance,
            };
        }
        let sum: f64 = model.components.iter().map(|c| c.weight).sum();
        if !sum.is_finite() || sum <= 0. {
            return Err("all components lost mass".into());
        }
        for c in &mut model.components {
            c.weight /= sum;
        }
        model.likelihood.push(model.log_likelihood(rows)?);
    }
    Ok(model)
}
pub fn run(_: &[String]) -> Result<()> {
    ch18::report(kmeans, gmm)
}
pub fn check() -> Result<()> {
    ch18::verify(kmeans, gmm)
}
