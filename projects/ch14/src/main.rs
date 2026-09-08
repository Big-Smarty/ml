//! k-nearest neighbors and Gaussian naive Bayes on one tiny three-class fixture.
#[derive(Clone, Copy, Debug)]
struct Point {
    x: [f64; 2],
    class: usize,
}

#[derive(Debug)]
struct Scaler {
    mean: [f64; 2],
    scale: [f64; 2],
}
impl Scaler {
    fn fit(data: &[Point]) -> Result<Self, &'static str> {
        if data.len() < 2 || data.iter().any(|p| p.x.iter().any(|v| !v.is_finite())) {
            return Err("scaling needs at least two finite points");
        }
        let mut mean = [0.0; 2];
        for p in data {
            for (sum, value) in mean.iter_mut().zip(p.x) {
                *sum += value;
            }
        }
        for m in &mut mean {
            *m /= data.len() as f64;
        }
        let mut scale = [0.0; 2];
        for p in data {
            for ((sum, value), center) in scale.iter_mut().zip(p.x).zip(mean) {
                *sum += (value - center).powi(2);
            }
        }
        for s in &mut scale {
            *s = (*s / data.len() as f64).sqrt();
            if *s == 0.0 {
                *s = 1.0;
            }
        }
        if mean.iter().chain(&scale).any(|v| !v.is_finite()) {
            return Err("scaling statistics overflowed; reduce feature magnitudes");
        }
        Ok(Self { mean, scale })
    }
    fn transform(&self, x: [f64; 2]) -> [f64; 2] {
        [
            (x[0] - self.mean[0]) / self.scale[0],
            (x[1] - self.mean[1]) / self.scale[1],
        ]
    }
}

fn knn(train: &[Point], query: [f64; 2], k: usize, classes: usize) -> Result<usize, &'static str> {
    if k == 0
        || k > train.len()
        || classes == 0
        || query.iter().any(|v| !v.is_finite())
        || train
            .iter()
            .any(|p| p.class >= classes || p.x.iter().any(|v| !v.is_finite()))
    {
        return Err("k, classes, and finite features must match the training data");
    }
    let mut distances: Vec<(f64, usize)> = train
        .iter()
        .map(|p| {
            (
                (p.x[0] - query[0]).powi(2) + (p.x[1] - query[1]).powi(2),
                p.class,
            )
        })
        .collect();
    if distances.iter().any(|(distance, _)| !distance.is_finite()) {
        return Err("distance overflowed; scale features before comparison");
    }
    // ponytail: a full sort is clear for six rows; use partial selection for large training sets.
    distances.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut votes = vec![0usize; classes];
    for &(_, class) in distances.iter().take(k) {
        votes[class] += 1;
    }
    Ok((0..classes)
        .max_by_key(|&c| (votes[c], std::cmp::Reverse(c)))
        .unwrap())
}

#[derive(Debug)]
struct GaussianNb {
    prior: Vec<f64>,
    mean: Vec<[f64; 2]>,
    variance: Vec<[f64; 2]>,
}
impl GaussianNb {
    fn fit(data: &[Point], classes: usize) -> Result<Self, &'static str> {
        if data.is_empty()
            || classes == 0
            || data
                .iter()
                .any(|p| p.class >= classes || p.x.iter().any(|v| !v.is_finite()))
        {
            return Err("Gaussian NB needs finite labelled training data");
        }
        let mut count = vec![0usize; classes];
        let mut mean = vec![[0.0; 2]; classes];
        for p in data {
            count[p.class] += 1;
            for (sum, value) in mean[p.class].iter_mut().zip(p.x) {
                *sum += value;
            }
        }
        if count.contains(&0) {
            return Err("every class needs a training point");
        }
        for (class, class_mean) in mean.iter_mut().enumerate() {
            for value in class_mean {
                *value /= count[class] as f64;
            }
        }
        let mut variance = vec![[0.0; 2]; classes];
        for p in data {
            for ((sum, value), center) in variance[p.class].iter_mut().zip(p.x).zip(mean[p.class]) {
                *sum += (value - center).powi(2);
            }
        }
        for (class, class_variance) in variance.iter_mut().enumerate() {
            for value in class_variance {
                *value = *value / count[class] as f64 + 1e-9;
            }
        }
        if mean
            .iter()
            .flatten()
            .chain(variance.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err("Gaussian statistics overflowed; scale features before fitting");
        }
        Ok(Self {
            prior: count
                .iter()
                .map(|&n| n as f64 / data.len() as f64)
                .collect(),
            mean,
            variance,
        })
    }
    fn predict(&self, x: [f64; 2]) -> Result<usize, &'static str> {
        if x.iter().any(|v| !v.is_finite()) {
            return Err("query coordinates must be finite");
        }
        let scores: Vec<_> = (0..self.prior.len())
            .map(|c| self.log_score(c, x))
            .collect();
        if scores.iter().any(|v| !v.is_finite()) {
            return Err("Gaussian score overflowed; scale features before prediction");
        }
        (0..scores.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .ok_or("Gaussian model has no classes")
    }
    fn log_score(&self, c: usize, x: [f64; 2]) -> f64 {
        let mut score = self.prior[c].ln();
        for (j, value) in x.into_iter().enumerate() {
            score += -0.5
                * ((2.0 * std::f64::consts::PI * self.variance[c][j]).ln()
                    + (value - self.mean[c][j]).powi(2) / self.variance[c][j]);
        }
        score
    }
}

fn main() -> Result<(), &'static str> {
    let raw = [
        Point {
            x: [1.0, 10.0],
            class: 0,
        },
        Point {
            x: [1.2, 11.0],
            class: 0,
        },
        Point {
            x: [3.0, 30.0],
            class: 1,
        },
        Point {
            x: [3.2, 29.0],
            class: 1,
        },
        Point {
            x: [5.0, 50.0],
            class: 2,
        },
        Point {
            x: [5.2, 49.0],
            class: 2,
        },
    ];
    let scaler = Scaler::fit(&raw)?;
    let train: Vec<Point> = raw
        .iter()
        .map(|p| Point {
            x: scaler.transform(p.x),
            class: p.class,
        })
        .collect();
    let query = scaler.transform([3.1, 31.0]);
    println!(
        "scaled query {query:?}; kNN class {}",
        knn(&train, query, 3, 3)?
    );
    println!(
        "Gaussian NB class {}",
        GaussianNb::fit(&train, 3)?.predict(query)?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_models_find_middle_cluster() {
        let d = [
            Point {
                x: [0., 0.],
                class: 0,
            },
            Point {
                x: [0.2, 0.1],
                class: 0,
            },
            Point {
                x: [2., 2.],
                class: 1,
            },
            Point {
                x: [2.2, 1.9],
                class: 1,
            },
            Point {
                x: [4., 4.],
                class: 2,
            },
            Point {
                x: [4.2, 3.9],
                class: 2,
            },
        ];
        assert_eq!(knn(&d, [2.1, 2.1], 3, 3).unwrap(), 1);
        assert_eq!(
            GaussianNb::fit(&d, 3).unwrap().predict([2.1, 2.1]).unwrap(),
            1
        );
        assert!(knn(&d, [0., 0.], 0, 3).is_err());
    }
    #[test]
    fn rejects_nonfinite_queries_and_overflow_instead_of_voting() {
        let point = Point {
            x: [0.0, 0.0],
            class: 0,
        };
        let model = GaussianNb::fit(&[point], 1).unwrap();
        assert!(model.predict([f64::NAN, 0.0]).is_err());
        assert!(model.predict([f64::MAX, 0.0]).is_err());
        assert!(knn(&[point], [f64::MAX, 0.0], 1, 1).is_err());
        let huge = Point {
            x: [f64::MAX, f64::MAX],
            class: 0,
        };
        assert!(Scaler::fit(&[huge, huge]).is_err());
        assert!(GaussianNb::fit(&[huge, huge], 1).is_err());
        let hand = GaussianNb {
            prior: vec![1.0],
            mean: vec![[2.0, 0.0]],
            variance: vec![[1.0, 1.0]],
        };
        assert!(
            (hand.log_score(0, [3.0, 0.0]) + (2.0 * std::f64::consts::PI).ln() + 0.5).abs() < 1e-12
        );
        let tied = [
            point,
            Point {
                x: [0.0, 0.0],
                class: 1,
            },
        ];
        assert_eq!(knn(&tied, [0.0, 0.0], 2, 2).unwrap(), 0);
    }
}
