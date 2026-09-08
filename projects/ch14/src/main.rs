//! k-nearest neighbors and Gaussian naive Bayes on one tiny three-class fixture.
#[derive(Clone, Copy, Debug)]
struct Point {
    features: [f64; 2],
    label: usize,
}

#[derive(Debug)]
struct Scaler {
    mean: [f64; 2],
    scale: [f64; 2],
}
impl Scaler {
    fn fit(train_data: &[Point]) -> Result<Self, &'static str> {
        if train_data.len() < 2
            || train_data
                .iter()
                .any(|point| point.features.iter().any(|value| !value.is_finite()))
        {
            return Err("scaling needs at least two finite points");
        }
        let mut mean = [0.0; 2];
        for point in train_data {
            for (sum, value) in mean.iter_mut().zip(point.features) {
                *sum += value;
            }
        }
        for m in &mut mean {
            *m /= train_data.len() as f64;
        }
        let mut scale = [0.0; 2];
        for point in train_data {
            for ((sum, value), center) in scale.iter_mut().zip(point.features).zip(mean) {
                *sum += (value - center).powi(2);
            }
        }
        for s in &mut scale {
            *s = (*s / train_data.len() as f64).sqrt();
            if *s == 0.0 {
                *s = 1.0;
            }
        }
        if mean.iter().chain(&scale).any(|v| !v.is_finite()) {
            return Err("scaling statistics overflowed; reduce feature magnitudes");
        }
        Ok(Self { mean, scale })
    }
    fn transform(&self, features: [f64; 2]) -> [f64; 2] {
        [
            (features[0] - self.mean[0]) / self.scale[0],
            (features[1] - self.mean[1]) / self.scale[1],
        ]
    }
}

fn squared_distance(features: [f64; 2], other_features: [f64; 2]) -> f64 {
    features
        .into_iter()
        .zip(other_features)
        .map(|(value, other_value)| (value - other_value).powi(2))
        .sum()
}

#[derive(Debug)]
struct Knn<'a> {
    train_data: &'a [Point],
    k: usize,
    classes: usize,
}
impl<'a> Knn<'a> {
    fn fit(train_data: &'a [Point], k: usize, classes: usize) -> Result<Self, &'static str> {
        if k == 0
            || k > train_data.len()
            || classes == 0
            || train_data.iter().any(|point| {
                point.label >= classes || point.features.iter().any(|value| !value.is_finite())
            })
        {
            return Err("k, classes, and finite features must match the training data");
        }
        Ok(Self {
            train_data,
            k,
            classes,
        })
    }

    fn predict(&self, features: [f64; 2]) -> Result<usize, &'static str> {
        if features.iter().any(|value| !value.is_finite()) {
            return Err("k, classes, and finite features must match the training data");
        }
        let mut distances: Vec<(f64, usize)> = self
            .train_data
            .iter()
            .map(|point| (squared_distance(point.features, features), point.label))
            .collect();
        if distances.iter().any(|(distance, _)| !distance.is_finite()) {
            return Err("distance overflowed; scale features before comparison");
        }
        // ponytail: a full sort is clear for six rows; use partial selection for large training sets.
        distances.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut votes = vec![0usize; self.classes];
        for &(_, label) in distances.iter().take(self.k) {
            votes[label] += 1;
        }
        (0..self.classes)
            .max_by_key(|&label| (votes[label], std::cmp::Reverse(label)))
            .ok_or("kNN model has no classes")
    }
}

#[derive(Debug)]
struct GaussianNb {
    priors: Vec<f64>,
    means: Vec<[f64; 2]>,
    variances: Vec<[f64; 2]>,
}
impl GaussianNb {
    fn fit(train_data: &[Point], classes: usize) -> Result<Self, &'static str> {
        if train_data.is_empty()
            || classes == 0
            || train_data.iter().any(|point| {
                point.label >= classes || point.features.iter().any(|value| !value.is_finite())
            })
        {
            return Err("Gaussian NB needs finite labelled training data");
        }
        let mut count = vec![0usize; classes];
        let mut means = vec![[0.0; 2]; classes];
        for point in train_data {
            count[point.label] += 1;
            for (sum, value) in means[point.label].iter_mut().zip(point.features) {
                *sum += value;
            }
        }
        if count.contains(&0) {
            return Err("every class needs a training point");
        }
        for (label, class_mean) in means.iter_mut().enumerate() {
            for value in class_mean {
                *value /= count[label] as f64;
            }
        }
        let mut variances = vec![[0.0; 2]; classes];
        for point in train_data {
            for ((sum, value), center) in variances[point.label]
                .iter_mut()
                .zip(point.features)
                .zip(means[point.label])
            {
                *sum += (value - center).powi(2);
            }
        }
        for (label, class_variance) in variances.iter_mut().enumerate() {
            for value in class_variance {
                *value = *value / count[label] as f64 + 1e-9;
            }
        }
        if means
            .iter()
            .flatten()
            .chain(variances.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err("Gaussian statistics overflowed; scale features before fitting");
        }
        Ok(Self {
            priors: count
                .iter()
                .map(|&n| n as f64 / train_data.len() as f64)
                .collect(),
            means,
            variances,
        })
    }
    fn predict(&self, features: [f64; 2]) -> Result<usize, &'static str> {
        if features.iter().any(|value| !value.is_finite()) {
            return Err("query coordinates must be finite");
        }
        let scores: Vec<_> = (0..self.priors.len())
            .map(|label| self.log_score(label, features))
            .collect();
        if scores.iter().any(|v| !v.is_finite()) {
            return Err("Gaussian score overflowed; scale features before prediction");
        }
        (0..scores.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .ok_or("Gaussian model has no classes")
    }
    fn log_score(&self, label: usize, features: [f64; 2]) -> f64 {
        let mut score = self.priors[label].ln();
        for (feature, value) in features.into_iter().enumerate() {
            score += -0.5
                * ((2.0 * std::f64::consts::PI * self.variances[label][feature]).ln()
                    + (value - self.means[label][feature]).powi(2)
                        / self.variances[label][feature]);
        }
        score
    }
}

fn main() -> Result<(), &'static str> {
    let raw_data = [
        Point {
            features: [1.0, 10.0],
            label: 0,
        },
        Point {
            features: [1.2, 11.0],
            label: 0,
        },
        Point {
            features: [3.0, 30.0],
            label: 1,
        },
        Point {
            features: [3.2, 29.0],
            label: 1,
        },
        Point {
            features: [5.0, 50.0],
            label: 2,
        },
        Point {
            features: [5.2, 49.0],
            label: 2,
        },
    ];
    let scaler = Scaler::fit(&raw_data)?;
    let train_data: Vec<Point> = raw_data
        .iter()
        .map(|point| Point {
            features: scaler.transform(point.features),
            label: point.label,
        })
        .collect();
    let query = scaler.transform([3.1, 31.0]);
    let knn = Knn::fit(&train_data, 3, 3)?;
    let gaussian_nb = GaussianNb::fit(&train_data, 3)?;
    println!("scaled query {query:?}; kNN class {}", knn.predict(query)?);
    println!("Gaussian NB class {}", gaussian_nb.predict(query)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_models_find_middle_cluster() {
        let train_data = [
            Point {
                features: [0., 0.],
                label: 0,
            },
            Point {
                features: [0.2, 0.1],
                label: 0,
            },
            Point {
                features: [2., 2.],
                label: 1,
            },
            Point {
                features: [2.2, 1.9],
                label: 1,
            },
            Point {
                features: [4., 4.],
                label: 2,
            },
            Point {
                features: [4.2, 3.9],
                label: 2,
            },
        ];
        assert_eq!(squared_distance([1.0, 2.0], [4.0, 6.0]), 25.0);
        assert_eq!(
            Knn::fit(&train_data, 3, 3)
                .unwrap()
                .predict([2.1, 2.1])
                .unwrap(),
            1
        );
        assert_eq!(
            GaussianNb::fit(&train_data, 3)
                .unwrap()
                .predict([2.1, 2.1])
                .unwrap(),
            1
        );
        assert!(Knn::fit(&train_data, 0, 3).is_err());
        let scaler = Scaler::fit(&[
            Point {
                features: [1.0, 10.0],
                label: 0,
            },
            Point {
                features: [3.0, 30.0],
                label: 1,
            },
        ])
        .unwrap();
        assert_eq!(scaler.transform([3.0, 30.0]), [1.0, 1.0]);
    }
    #[test]
    fn rejects_nonfinite_queries_and_overflow_instead_of_voting() {
        let point = Point {
            features: [0.0, 0.0],
            label: 0,
        };
        let invalid_label = Point {
            features: [0.0, 0.0],
            label: 1,
        };
        assert!(Knn::fit(&[invalid_label], 1, 1).is_err());
        let model = GaussianNb::fit(&[point], 1).unwrap();
        assert!(model.predict([f64::NAN, 0.0]).is_err());
        assert!(model.predict([f64::MAX, 0.0]).is_err());
        assert!(Knn::fit(&[point], 1, 1)
            .unwrap()
            .predict([f64::MAX, 0.0])
            .is_err());
        let huge = Point {
            features: [f64::MAX, f64::MAX],
            label: 0,
        };
        assert!(Scaler::fit(&[huge, huge]).is_err());
        assert!(GaussianNb::fit(&[huge, huge], 1).is_err());
        let hand = GaussianNb {
            priors: vec![1.0],
            means: vec![[2.0, 0.0]],
            variances: vec![[1.0, 1.0]],
        };
        assert!(
            (hand.log_score(0, [3.0, 0.0]) + (2.0 * std::f64::consts::PI).ln() + 0.5).abs() < 1e-12
        );
        let tied = [
            point,
            Point {
                features: [0.0, 0.0],
                label: 1,
            },
        ];
        assert_eq!(
            Knn::fit(&tied, 2, 2).unwrap().predict([0.0, 0.0]).unwrap(),
            0
        );
    }
}
