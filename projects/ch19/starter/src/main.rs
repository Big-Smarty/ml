//! Reproducible cross-validation and bounded search on a bundled tabular fixture.

#[derive(Clone, Copy, Debug)]
struct Row {
    id: u64,
    x: [f64; 2],
    y: u8,
}

fn validate(rows: &[Row]) -> Result<(), &'static str> {
    if rows.len() < 2 {
        return Err("evaluation requires at least two rows");
    }
    if rows
        .iter()
        .any(|r| r.y > 1 || r.x.iter().any(|x| !x.is_finite()))
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
    fn fit(rows: &[Row]) -> Result<Self, &'static str> {
        validate(rows)?;
        let mut mean = [0.0; 2];
        for r in rows {
            mean[0] += r.x[0];
            mean[1] += r.x[1];
        }
        mean[0] /= rows.len() as f64;
        mean[1] /= rows.len() as f64;
        let mut variance = [0.0; 2];
        for r in rows {
            variance[0] += (r.x[0] - mean[0]).powi(2);
            variance[1] += (r.x[1] - mean[1]).powi(2);
        }
        let mut scale = [
            (variance[0] / rows.len() as f64).sqrt(),
            (variance[1] / rows.len() as f64).sqrt(),
        ];
        for s in &mut scale {
            if *s == 0.0 {
                *s = 1.0;
            }
        }
        if mean.iter().chain(&scale).any(|x| !x.is_finite()) {
            return Err("scaling overflow; rescale the raw features");
        }
        Ok(Self { mean, scale })
    }
    fn apply(self, x: [f64; 2]) -> [f64; 2] {
        [
            (x[0] - self.mean[0]) / self.scale[0],
            (x[1] - self.mean[1]) / self.scale[1],
        ]
    }
}

#[derive(Clone, Copy, Debug)]
struct Config {
    rate: f64,
    l2: f64,
    epochs: usize,
}

#[derive(Clone, Copy, Debug)]
struct Logistic {
    w: [f64; 2],
    b: f64,
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

impl Logistic {
    fn fit(rows: &[Row], config: Config) -> Result<Self, &'static str> {
        validate(rows)?;
        if config.epochs == 0
            || !config.rate.is_finite()
            || config.rate <= 0.0
            || !config.l2.is_finite()
            || config.l2 < 0.0
        {
            return Err("training settings must be finite and in range");
        }
        let scaler = Scaler::fit(rows)?;
        let mut model = Self {
            w: [0.0; 2],
            b: 0.0,
            scaler,
        };
        for _ in 0..config.epochs {
            let mut dw = [0.0; 2];
            let mut db = 0.0;
            for r in rows {
                let x = scaler.apply(r.x);
                let error = sigmoid(model.w[0] * x[0] + model.w[1] * x[1] + model.b) - r.y as f64;
                dw[0] += error * x[0];
                dw[1] += error * x[1];
                db += error;
            }
            let n = rows.len() as f64;
            model.w[0] -= config.rate * (dw[0] / n + config.l2 * model.w[0]);
            model.w[1] -= config.rate * (dw[1] / n + config.l2 * model.w[1]);
            model.b -= config.rate * db / n;
        }
        if model.w.iter().any(|x| !x.is_finite()) || !model.b.is_finite() {
            return Err("training diverged; reduce the learning rate");
        }
        Ok(model)
    }
    fn probability(self, x: [f64; 2]) -> f64 {
        let x = self.scaler.apply(x);
        sigmoid(self.w[0] * x[0] + self.w[1] * x[1] + self.b)
    }
    fn predict(self, x: [f64; 2]) -> u8 {
        u8::from(self.probability(x) >= 0.5)
    }
}

fn accuracy(rows: &[Row], predict: impl Fn([f64; 2]) -> u8) -> f64 {
    rows.iter().filter(|r| predict(r.x) == r.y).count() as f64 / rows.len() as f64
}

fn majority_label(rows: &[Row]) -> u8 {
    u8::from(rows.iter().filter(|r| r.y == 1).count() * 2 >= rows.len())
}

const DEV: [Row; 24] = [
    Row {
        id: 1,
        x: [-2.0, -1.0],
        y: 0,
    },
    Row {
        id: 2,
        x: [-1.8, -0.6],
        y: 0,
    },
    Row {
        id: 3,
        x: [-1.5, -1.4],
        y: 0,
    },
    Row {
        id: 4,
        x: [-1.2, -0.4],
        y: 0,
    },
    Row {
        id: 5,
        x: [-1.0, -1.2],
        y: 0,
    },
    Row {
        id: 6,
        x: [-0.8, -0.2],
        y: 0,
    },
    Row {
        id: 7,
        x: [-0.6, -0.9],
        y: 0,
    },
    Row {
        id: 8,
        x: [-0.4, -0.3],
        y: 0,
    },
    Row {
        id: 9,
        x: [-0.2, -0.8],
        y: 0,
    },
    Row {
        id: 10,
        x: [0.1, -0.7],
        y: 0,
    },
    Row {
        id: 11,
        x: [0.2, -0.3],
        y: 0,
    },
    Row {
        id: 12,
        x: [0.5, -0.9],
        y: 0,
    },
    Row {
        id: 13,
        x: [-0.4, 1.1],
        y: 1,
    },
    Row {
        id: 14,
        x: [-0.1, 0.5],
        y: 1,
    },
    Row {
        id: 15,
        x: [0.2, 0.8],
        y: 1,
    },
    Row {
        id: 16,
        x: [0.4, 0.3],
        y: 1,
    },
    Row {
        id: 17,
        x: [0.6, 1.3],
        y: 1,
    },
    Row {
        id: 18,
        x: [0.8, 0.5],
        y: 1,
    },
    Row {
        id: 19,
        x: [1.0, 1.0],
        y: 1,
    },
    Row {
        id: 20,
        x: [1.2, 0.4],
        y: 1,
    },
    Row {
        id: 21,
        x: [1.4, 1.4],
        y: 1,
    },
    Row {
        id: 22,
        x: [1.6, 0.7],
        y: 1,
    },
    Row {
        id: 23,
        x: [1.8, 1.2],
        y: 1,
    },
    Row {
        id: 24,
        x: [2.0, 0.6],
        y: 1,
    },
];

#[cfg(test)]
fn pooled_accuracy(fold_counts: &[(usize, usize)]) -> f64 {
    let _ = fold_counts;
    // TODO: sum correct and total across folds, then divide once.
    todo!("compute pooled accuracy")
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (fit, valid) = DEV.split_at(18);
    let baseline = majority_label(fit);
    let config = Config {
        rate: 0.1,
        l2: 0.01,
        epochs: 100,
    };
    let model = Logistic::fit(fit, config)?;
    println!("Starting checkpoint: one fixed development split, no search yet.");
    println!(
        "fit IDs {:?}; validation IDs {:?}",
        fit.iter().map(|r| r.id).collect::<Vec<_>>(),
        valid.iter().map(|r| r.id).collect::<Vec<_>>()
    );
    println!(
        "majority validation {:.1}%; logistic validation {:.1}%",
        100.0 * accuracy(valid, |_| baseline),
        100.0 * accuracy(valid, |x| model.predict(x))
    );
    println!("Complete pooled_accuracy, then replace this single split with stratified folds before searching.");
    Ok(())
}
#[test]
fn unequal_folds_are_weighted_by_rows() {
    assert!((pooled_accuracy(&[(8, 10), (1, 2)]) - 0.75).abs() < 1e-12);
}
