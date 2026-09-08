//! Reproducible cross-validation and bounded search on a bundled tabular fixture.

#[derive(Clone, Copy, Debug)]
struct Row {
    id: u64,
    features: [f64; 2],
    label: u8,
}

fn validate(rows: &[Row]) -> Result<(), &'static str> {
    if rows.len() < 2 {
        return Err("evaluation requires at least two rows");
    }
    if rows
        .iter()
        .any(|row| row.label > 1 || row.features.iter().any(|value| !value.is_finite()))
    {
        return Err("features must be finite and labels must be 0 or 1");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Scaler {
    mean: [f64; 2],
    scale: [f64; 2],
}

impl Scaler {
    fn fit(train_data: &[Row]) -> Result<Self, &'static str> {
        validate(train_data)?;
        let mut mean = [0.0; 2];
        for row in train_data {
            mean[0] += row.features[0];
            mean[1] += row.features[1];
        }
        mean[0] /= train_data.len() as f64;
        mean[1] /= train_data.len() as f64;
        let mut variance = [0.0; 2];
        for row in train_data {
            variance[0] += (row.features[0] - mean[0]).powi(2);
            variance[1] += (row.features[1] - mean[1]).powi(2);
        }
        let mut scale = [
            (variance[0] / train_data.len() as f64).sqrt(),
            (variance[1] / train_data.len() as f64).sqrt(),
        ];
        for s in &mut scale {
            if *s == 0.0 {
                *s = 1.0;
            }
        }
        if mean.iter().chain(&scale).any(|value| !value.is_finite()) {
            return Err("scaling overflow; rescale the raw features");
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

#[derive(Clone, Copy, Debug)]
struct Config {
    learning_rate: f64,
    l2: f64,
    epochs: usize,
}

#[derive(Clone, Copy, Debug)]
struct LogisticModel {
    weights: [f64; 2],
    bias: f64,
    scaler: Scaler,
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

impl LogisticModel {
    fn fit(train_data: &[Row], config: Config) -> Result<Self, &'static str> {
        validate(train_data)?;
        if config.epochs == 0
            || !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
            || !config.l2.is_finite()
            || config.l2 < 0.0
        {
            return Err("training settings must be finite and in range");
        }
        let scaler = Scaler::fit(train_data)?;
        let mut model = Self {
            weights: [0.0; 2],
            bias: 0.0,
            scaler,
        };
        for _ in 0..config.epochs {
            let mut weight_gradient = [0.0; 2];
            let mut bias_gradient = 0.0;
            for row in train_data {
                let features = scaler.transform(row.features);
                let error = sigmoid(
                    model.weights[0] * features[0] + model.weights[1] * features[1] + model.bias,
                ) - row.label as f64;
                weight_gradient[0] += error * features[0];
                weight_gradient[1] += error * features[1];
                bias_gradient += error;
            }
            let n = train_data.len() as f64;
            model.weights[0] -=
                config.learning_rate * (weight_gradient[0] / n + config.l2 * model.weights[0]);
            model.weights[1] -=
                config.learning_rate * (weight_gradient[1] / n + config.l2 * model.weights[1]);
            model.bias -= config.learning_rate * bias_gradient / n;
        }
        if model.weights.iter().any(|value| !value.is_finite()) || !model.bias.is_finite() {
            return Err("training diverged; reduce the learning rate");
        }
        Ok(model)
    }
    fn probability(&self, features: [f64; 2]) -> f64 {
        let features = self.scaler.transform(features);
        sigmoid(self.weights[0] * features[0] + self.weights[1] * features[1] + self.bias)
    }
    fn predict(&self, features: [f64; 2]) -> u8 {
        u8::from(self.probability(features) >= 0.5)
    }
}

fn accuracy(rows: &[Row], predict: impl Fn([f64; 2]) -> u8) -> f64 {
    rows.iter()
        .filter(|row| predict(row.features) == row.label)
        .count() as f64
        / rows.len() as f64
}

fn majority_label(rows: &[Row]) -> u8 {
    u8::from(rows.iter().filter(|row| row.label == 1).count() * 2 >= rows.len())
}

const DEV: [Row; 24] = [
    Row {
        id: 1,
        features: [-2.0, -1.0],
        label: 0,
    },
    Row {
        id: 2,
        features: [-1.8, -0.6],
        label: 0,
    },
    Row {
        id: 3,
        features: [-1.5, -1.4],
        label: 0,
    },
    Row {
        id: 4,
        features: [-1.2, -0.4],
        label: 0,
    },
    Row {
        id: 5,
        features: [-1.0, -1.2],
        label: 0,
    },
    Row {
        id: 6,
        features: [-0.8, -0.2],
        label: 0,
    },
    Row {
        id: 7,
        features: [-0.6, -0.9],
        label: 0,
    },
    Row {
        id: 8,
        features: [-0.4, -0.3],
        label: 0,
    },
    Row {
        id: 9,
        features: [-0.2, -0.8],
        label: 0,
    },
    Row {
        id: 10,
        features: [0.1, -0.7],
        label: 0,
    },
    Row {
        id: 11,
        features: [0.2, -0.3],
        label: 0,
    },
    Row {
        id: 12,
        features: [0.5, -0.9],
        label: 0,
    },
    Row {
        id: 13,
        features: [-0.4, 1.1],
        label: 1,
    },
    Row {
        id: 14,
        features: [-0.1, 0.5],
        label: 1,
    },
    Row {
        id: 15,
        features: [0.2, 0.8],
        label: 1,
    },
    Row {
        id: 16,
        features: [0.4, 0.3],
        label: 1,
    },
    Row {
        id: 17,
        features: [0.6, 1.3],
        label: 1,
    },
    Row {
        id: 18,
        features: [0.8, 0.5],
        label: 1,
    },
    Row {
        id: 19,
        features: [1.0, 1.0],
        label: 1,
    },
    Row {
        id: 20,
        features: [1.2, 0.4],
        label: 1,
    },
    Row {
        id: 21,
        features: [1.4, 1.4],
        label: 1,
    },
    Row {
        id: 22,
        features: [1.6, 0.7],
        label: 1,
    },
    Row {
        id: 23,
        features: [1.8, 1.2],
        label: 1,
    },
    Row {
        id: 24,
        features: [2.0, 0.6],
        label: 1,
    },
];

#[cfg(test)]
fn pooled_accuracy(fold_counts: &[(usize, usize)]) -> f64 {
    let _ = fold_counts;
    // TODO: sum correct and total across folds, then divide once.
    todo!("compute pooled accuracy")
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (train_data, validation_data) = DEV.split_at(18);
    let baseline = majority_label(train_data);
    let config = Config {
        learning_rate: 0.1,
        l2: 0.01,
        epochs: 100,
    };
    let model = LogisticModel::fit(train_data, config)?;
    println!("Starting checkpoint: one fixed development split, no search yet.");
    println!(
        "fit IDs {:?}; validation IDs {:?}",
        train_data.iter().map(|row| row.id).collect::<Vec<_>>(),
        validation_data.iter().map(|row| row.id).collect::<Vec<_>>()
    );
    println!(
        "majority validation {:.1}%; logistic validation {:.1}%",
        100.0 * accuracy(validation_data, |_| baseline),
        100.0 * accuracy(validation_data, |features| model.predict(features))
    );
    println!("Complete pooled_accuracy, then replace this single split with stratified folds before searching.");
    Ok(())
}
#[test]
fn unequal_folds_are_weighted_by_rows() {
    assert!((pooled_accuracy(&[(8, 10), (1, 2)]) - 0.75).abs() < 1e-12);
}
