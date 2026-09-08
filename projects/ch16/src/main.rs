//! Linear hinge-loss training and an RBF-kernel perceptron, using only std.

#[derive(Clone, Copy, Debug)]
struct Point {
    features: [f64; 2],
    label: f64,
}

fn validate(data: &[Point]) -> Result<(), &'static str> {
    if data.is_empty() {
        return Err("training requires at least one point");
    }
    if data.iter().any(|point| {
        !point.features[0].is_finite()
            || !point.features[1].is_finite()
            || !point.label.is_finite()
            || point.label.abs() != 1.0
    }) {
        return Err("features must be finite and labels must be -1 or +1");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct LinearSvm {
    weights: [f64; 2],
    bias: f64,
}

fn hinge_loss(score: f64, label: f64) -> f64 {
    (1.0 - label * score).max(0.0)
}

impl LinearSvm {
    fn score(&self, features: [f64; 2]) -> f64 {
        self.weights[0] * features[0] + self.weights[1] * features[1] + self.bias
    }

    fn predict(&self, features: [f64; 2]) -> f64 {
        if self.score(features) >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    fn objective(&self, data: &[Point], lambda: f64) -> Result<f64, &'static str> {
        validate(data)?;
        if !lambda.is_finite() || lambda < 0.0 {
            return Err("lambda must be finite and non-negative");
        }
        if self.weights.iter().any(|weight| !weight.is_finite()) || !self.bias.is_finite() {
            return Err("model parameters must be finite");
        }
        let hinge = data
            .iter()
            .map(|point| hinge_loss(self.score(point.features), point.label))
            .sum::<f64>()
            / data.len() as f64;
        let objective =
            0.5 * lambda * (self.weights[0] * self.weights[0] + self.weights[1] * self.weights[1])
                + hinge;
        if !objective.is_finite() {
            return Err("objective overflow; rescale data or reduce the learning rate");
        }
        Ok(objective)
    }

    fn fit(
        data: &[Point],
        epochs: usize,
        learning_rate: f64,
        lambda: f64,
    ) -> Result<Self, &'static str> {
        validate(data)?;
        if epochs == 0
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
            || !lambda.is_finite()
            || lambda < 0.0
        {
            return Err("epochs must be positive, learning rate must be finite and positive, and lambda must be finite and non-negative");
        }
        let mut model = Self {
            weights: [0.0; 2],
            bias: 0.0,
        };
        for _ in 0..epochs {
            for point in data {
                let margin = point.label * model.score(point.features);
                model.weights[0] *= 1.0 - learning_rate * lambda;
                model.weights[1] *= 1.0 - learning_rate * lambda;
                if margin < 1.0 {
                    model.weights[0] += learning_rate * point.label * point.features[0];
                    model.weights[1] += learning_rate * point.label * point.features[1];
                    model.bias += learning_rate * point.label;
                }
            }
        }
        model.objective(data, lambda)?;
        Ok(model)
    }
}

fn squared_distance(features: [f64; 2], other_features: [f64; 2]) -> f64 {
    features
        .into_iter()
        .zip(other_features)
        .map(|(value, other_value)| (value - other_value).powi(2))
        .sum()
}

fn rbf(a: [f64; 2], b: [f64; 2], gamma: f64) -> f64 {
    (-gamma * squared_distance(a, b)).exp()
}

struct KernelClassifier<'a> {
    data: &'a [Point],
    alpha: Vec<f64>,
    gamma: f64,
}

impl<'a> KernelClassifier<'a> {
    fn fit(data: &'a [Point], epochs: usize, gamma: f64) -> Result<Self, &'static str> {
        validate(data)?;
        if epochs == 0 || !gamma.is_finite() || gamma <= 0.0 {
            return Err("epochs and a finite positive gamma are required");
        }
        let mut model = Self {
            data,
            alpha: vec![0.0; data.len()],
            gamma,
        };
        for _ in 0..epochs {
            for (i, point) in data.iter().enumerate() {
                if point.label * model.score(point.features) <= 0.0 {
                    model.alpha[i] += 1.0;
                }
            }
        }
        Ok(model)
    }

    fn score(&self, features: [f64; 2]) -> f64 {
        self.data
            .iter()
            .zip(&self.alpha)
            .map(|(point, &alpha)| alpha * point.label * rbf(point.features, features, self.gamma))
            .sum()
    }

    fn predict(&self, features: [f64; 2]) -> f64 {
        if self.score(features) >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    fn support_count(&self) -> usize {
        self.alpha.iter().filter(|&&a| a > 0.0).count()
    }
}

fn accuracy(data: &[Point], predict: impl Fn([f64; 2]) -> f64) -> f64 {
    data.iter()
        .filter(|point| predict(point.features) == point.label)
        .count() as f64
        / data.len() as f64
}

const LINEAR: [Point; 8] = [
    Point {
        features: [-2.0, -1.0],
        label: -1.0,
    },
    Point {
        features: [-1.5, 0.0],
        label: -1.0,
    },
    Point {
        features: [-1.0, -0.5],
        label: -1.0,
    },
    Point {
        features: [-0.5, -1.5],
        label: -1.0,
    },
    Point {
        features: [0.5, 1.5],
        label: 1.0,
    },
    Point {
        features: [1.0, 0.5],
        label: 1.0,
    },
    Point {
        features: [1.5, 0.0],
        label: 1.0,
    },
    Point {
        features: [2.0, 1.0],
        label: 1.0,
    },
];

const XOR: [Point; 8] = [
    Point {
        features: [-1.2, -0.8],
        label: 1.0,
    },
    Point {
        features: [-0.8, -1.2],
        label: 1.0,
    },
    Point {
        features: [0.8, 1.2],
        label: 1.0,
    },
    Point {
        features: [1.2, 0.8],
        label: 1.0,
    },
    Point {
        features: [-1.2, 0.8],
        label: -1.0,
    },
    Point {
        features: [-0.8, 1.2],
        label: -1.0,
    },
    Point {
        features: [0.8, -1.2],
        label: -1.0,
    },
    Point {
        features: [1.2, -0.8],
        label: -1.0,
    },
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let linear = LinearSvm::fit(&LINEAR, 80, 0.05, 0.01)?;
    println!("linear SVM: w={:?}, b={:.3}", linear.weights, linear.bias);
    println!(
        "linear fixture accuracy: {:.1}%",
        100.0 * accuracy(&LINEAR, |features| linear.predict(features))
    );
    println!(
        "hinge-plus-penalty objective: {:.4}",
        linear.objective(&LINEAR, 0.01)?
    );

    let straight_on_xor = LinearSvm::fit(&XOR, 80, 0.05, 0.01)?;
    let kernel = KernelClassifier::fit(&XOR, 12, 1.0)?;
    println!(
        "linear accuracy on curved/XOR fixture: {:.1}%",
        100.0 * accuracy(&XOR, |features| straight_on_xor.predict(features))
    );
    println!(
        "RBF kernel accuracy: {:.1}% ({} support points)",
        100.0 * accuracy(&XOR, |features| kernel.predict(features)),
        kernel.support_count()
    );
    println!("Synthetic fixtures verify mechanics; they do not estimate real-world accuracy.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_divergent_finite_settings() {
        let data = [Point {
            features: [2.0, 1.0],
            label: 1.0,
        }];
        assert!(LinearSvm::fit(&data, 2, f64::MAX, 1.0).is_err());
    }

    #[test]
    fn hinge_and_linear_fit_match_hand_checks() -> Result<(), &'static str> {
        let zero = LinearSvm {
            weights: [0.0; 2],
            bias: 0.0,
        };
        assert!((zero.objective(&LINEAR, 0.01)? - 1.0).abs() < 1e-12);

        let hand_model = LinearSvm {
            weights: [0.5, 1.0],
            bias: -0.5,
        };
        let hand_point = [Point {
            features: [2.0, 1.0],
            label: 1.0,
        }];
        assert_eq!(hand_model.score(hand_point[0].features), 1.5);
        assert_eq!(hand_model.predict(hand_point[0].features), 1.0);
        assert!((hand_model.objective(&hand_point, 0.1)? - 0.0625).abs() < 1e-12);

        let model = LinearSvm::fit(&LINEAR, 80, 0.05, 0.01)?;
        assert_eq!(accuracy(&LINEAR, |features| model.predict(features)), 1.0);
        Ok(())
    }

    #[test]
    fn kernel_handles_xor_where_a_line_does_not() -> Result<(), &'static str> {
        let linear = LinearSvm::fit(&XOR, 80, 0.05, 0.01)?;
        assert!(accuracy(&XOR, |features| linear.predict(features)) <= 0.75);
        let kernel = KernelClassifier::fit(&XOR, 12, 1.0)?;
        assert_eq!(accuracy(&XOR, |features| kernel.predict(features)), 1.0);
        assert!(kernel.support_count() > 0);
        Ok(())
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert!(LinearSvm::fit(&[], 1, 0.1, 0.0).is_err());
        assert!(LinearSvm::fit(
            &[Point {
                features: [0.0, 0.0],
                label: 0.0,
            }],
            1,
            0.1,
            0.0
        )
        .is_err());
        assert!(KernelClassifier::fit(&LINEAR, 1, 0.0).is_err());
    }
}
