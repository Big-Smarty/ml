//! Deterministic k-means, diagonal Gaussian-mixture EM, and density anomaly scores.

type Point = [f64; 2];

fn validate(data: &[Point]) -> Result<(), &'static str> {
    if data.len() < 2 {
        return Err("clustering requires at least two points");
    }
    if data.iter().flatten().any(|x| !x.is_finite()) {
        return Err("all features must be finite");
    }
    Ok(())
}

fn distance2(a: Point, b: Point) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

#[derive(Debug)]
struct KMeans {
    centers: Vec<Point>,
    assignments: Vec<usize>,
    inertia: f64,
}

impl KMeans {
    fn fit(data: &[Point], k: usize, steps: usize) -> Result<Self, &'static str> {
        validate(data)?;
        if k == 0 || k > data.len() || steps == 0 {
            return Err("k and steps must be in range");
        }
        let mut centers: Vec<_> = (0..k)
            .map(|j| data[j * (data.len() - 1) / (k - 1).max(1)])
            .collect();
        let mut assignments = vec![0; data.len()];
        for _ in 0..steps {
            for (i, &x) in data.iter().enumerate() {
                assignments[i] = centers
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| distance2(x, **a).total_cmp(&distance2(x, **b)))
                    .map(|(j, _)| j)
                    .unwrap();
            }
            let mut sums = vec![[0.0; 2]; k];
            let mut counts = vec![0usize; k];
            for (&x, &j) in data.iter().zip(&assignments) {
                sums[j][0] += x[0];
                sums[j][1] += x[1];
                counts[j] += 1;
            }
            for j in 0..k {
                if counts[j] > 0 {
                    centers[j] = [sums[j][0] / counts[j] as f64, sums[j][1] / counts[j] as f64];
                }
                // ponytail: retain an empty center; add farthest-point reseeding for adversarial starts.
            }
        }
        for (i, &x) in data.iter().enumerate() {
            assignments[i] = centers
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| distance2(x, **a).total_cmp(&distance2(x, **b)))
                .map(|(j, _)| j)
                .unwrap();
        }
        let inertia: f64 = data
            .iter()
            .zip(&assignments)
            .map(|(&x, &j)| distance2(x, centers[j]))
            .sum();
        if !inertia.is_finite() || centers.iter().flatten().any(|x| !x.is_finite()) {
            return Err("clustering overflow; rescale the input coordinates");
        }
        Ok(Self {
            centers,
            assignments,
            inertia,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Component {
    weight: f64,
    mean: Point,
    variance: Point,
}

#[derive(Debug)]
struct GaussianMixture {
    components: Vec<Component>,
}

fn log_gaussian(x: Point, c: Component) -> f64 {
    let mut log_prob = 0.0;
    for (d, value) in x.iter().enumerate() {
        log_prob += (2.0 * std::f64::consts::PI).ln()
            + c.variance[d].ln()
            + (*value - c.mean[d]).powi(2) / c.variance[d];
    }
    -0.5 * log_prob
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let m = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    m + values.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
}

impl GaussianMixture {
    fn fit(
        data: &[Point],
        k: usize,
        steps: usize,
        variance_floor: f64,
    ) -> Result<Self, &'static str> {
        validate(data)?;
        if k == 0
            || k > data.len()
            || steps == 0
            || !variance_floor.is_finite()
            || variance_floor <= 0.0
        {
            return Err("valid k, steps, and positive variance floor are required");
        }
        let seeds = KMeans::fit(data, k, 8)?;
        let mut model = Self {
            components: seeds
                .centers
                .iter()
                .map(|&mean| Component {
                    weight: 1.0 / k as f64,
                    mean,
                    variance: [1.0, 1.0],
                })
                .collect(),
        };
        for _ in 0..steps {
            let responsibilities: Vec<Vec<f64>> =
                data.iter().map(|&x| model.responsibilities(x)).collect();
            if responsibilities.iter().flatten().any(|r| !r.is_finite()) {
                return Err("non-finite responsibilities; rescale the inputs");
            }
            for j in 0..k {
                let mass: f64 = responsibilities.iter().map(|r| r[j]).sum();
                if mass <= 1e-12 {
                    model.components[j].weight = 0.0;
                    continue;
                }
                let mut mean = [0.0; 2];
                for (&x, r) in data.iter().zip(&responsibilities) {
                    mean[0] += r[j] * x[0];
                    mean[1] += r[j] * x[1];
                }
                mean[0] /= mass;
                mean[1] /= mass;
                let mut variance = [0.0; 2];
                for (&x, r) in data.iter().zip(&responsibilities) {
                    variance[0] += r[j] * (x[0] - mean[0]).powi(2);
                    variance[1] += r[j] * (x[1] - mean[1]).powi(2);
                }
                variance[0] = (variance[0] / mass).max(variance_floor);
                variance[1] = (variance[1] / mass).max(variance_floor);
                model.components[j] = Component {
                    weight: mass / data.len() as f64,
                    mean,
                    variance,
                };
            }
            let weight_sum: f64 = model.components.iter().map(|c| c.weight).sum();
            if !weight_sum.is_finite()
                || weight_sum <= 0.0
                || model
                    .components
                    .iter()
                    .any(|c| c.mean.iter().chain(&c.variance).any(|x| !x.is_finite()))
            {
                return Err("non-finite mixture update; rescale the inputs");
            }
            for c in &mut model.components {
                c.weight /= weight_sum;
            }
        }
        Ok(model)
    }

    fn log_terms(&self, x: Point) -> Vec<f64> {
        self.components
            .iter()
            .map(|&c| c.weight.ln() + log_gaussian(x, c))
            .collect()
    }

    fn log_density(&self, x: Point) -> f64 {
        log_sum_exp(&self.log_terms(x))
    }

    fn responsibilities(&self, x: Point) -> Vec<f64> {
        let terms = self.log_terms(x);
        let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<_> = terms.iter().map(|v| (v - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        weights.iter().map(|w| w / total).collect()
    }

    fn anomaly_score(&self, x: Point) -> f64 {
        -self.log_density(x)
    }

    fn log_likelihood(&self, data: &[Point]) -> f64 {
        data.iter().map(|&x| self.log_density(x)).sum()
    }
}

const DATA: [Point; 12] = [
    [-2.3, -1.8],
    [-2.0, -2.1],
    [-1.8, -2.2],
    [-2.1, -1.7],
    [-1.7, -1.9],
    [-2.2, -2.3],
    [2.2, 1.8],
    [2.0, 2.2],
    [1.8, 1.9],
    [2.3, 2.1],
    [1.7, 2.3],
    [2.1, 1.7],
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let kmeans = KMeans::fit(&DATA, 2, 12)?;
    println!("k-means centers: {:?}", kmeans.centers);
    println!(
        "assignments: {:?}; inertia: {:.3}",
        kmeans.assignments, kmeans.inertia
    );
    let mixture = GaussianMixture::fit(&DATA, 2, 20, 1e-3)?;
    println!("GMM components: {:?}", mixture.components);
    println!(
        "training log likelihood: {:.3}",
        mixture.log_likelihood(&DATA)
    );
    for x in [[-2.0, -2.0], [0.0, 0.0], [8.0, 8.0]] {
        println!(
            "point {x:?}: responsibilities {:?}, anomaly score {:.3}",
            mixture.responsibilities(x),
            mixture.anomaly_score(x)
        );
    }
    println!("Thresholds require held-out normal data; these scores are demonstrations only.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_survives_large_common_log_offset() {
        let component = Component {
            weight: 0.5,
            mean: [0.0; 2],
            variance: [1.0; 2],
        };
        let model = GaussianMixture {
            components: vec![component; 2],
        };
        assert_eq!(model.responsibilities([1e8; 2]), vec![0.5, 0.5]);
    }
    #[test]
    fn assignments_use_final_centers_and_overflow_is_rejected() {
        let rows = [[0.0, 0.0], [4.0, 0.0], [5.0, 0.0], [5.1, 0.0], [10.0, 0.0]];
        let model = KMeans::fit(&rows, 2, 1).unwrap();
        assert_eq!(model.assignments[3], 0);
        assert!(KMeans::fit(&[[f64::MAX, 0.0], [-f64::MAX, 0.0]], 1, 1).is_err());
    }

    #[test]
    fn kmeans_finds_two_fixture_groups() -> Result<(), &'static str> {
        let first = KMeans::fit(&DATA, 2, 1)?;
        let model = KMeans::fit(&DATA, 2, 12)?;
        assert!(model.centers.iter().any(|c| c[0] < -1.5));
        assert!(model.centers.iter().any(|c| c[0] > 1.5));
        assert!(model.inertia < 3.0);
        assert!(model.inertia <= first.inertia + 1e-12);
        Ok(())
    }

    #[test]
    fn em_probabilities_normalize_and_outlier_scores_higher() -> Result<(), &'static str> {
        let first = GaussianMixture::fit(&DATA, 2, 1, 1e-3)?;
        let model = GaussianMixture::fit(&DATA, 2, 20, 1e-3)?;
        let r = model.responsibilities([0.0, 0.0]);
        assert!((r.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(model.anomaly_score([8.0, 8.0]) > model.anomaly_score([-2.0, -2.0]));
        assert!(model
            .components
            .iter()
            .all(|c| c.variance.iter().all(|&v| v >= 1e-3)));
        assert!((model.components.iter().map(|c| c.weight).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(model.log_likelihood(&DATA) + 1e-10 >= first.log_likelihood(&DATA));
        Ok(())
    }

    #[test]
    fn invalid_settings_are_rejected() {
        assert!(KMeans::fit(&[], 2, 2).is_err());
        assert!(GaussianMixture::fit(&DATA, 2, 2, 0.0).is_err());
    }
}
