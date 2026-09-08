//! Compare learning rates, minibatches, and L2 regularization on noisy regression.

const TRAIN: [(f64, f64); 12] = [
    (-3.0, -5.2),
    (-2.5, -3.7),
    (-2.0, -3.3),
    (-1.5, -1.7),
    (-1.0, -1.2),
    (-0.5, 0.4),
    (0.0, 0.8),
    (0.5, 2.2),
    (1.0, 2.7),
    (1.5, 4.4),
    (2.0, 4.8),
    (2.5, 6.4),
];
const VALIDATION: [(f64, f64); 6] = [
    (-2.75, -4.4),
    (-1.25, -1.4),
    (-0.25, 0.6),
    (0.75, 2.5),
    (1.75, 4.6),
    (2.75, 6.3),
];

#[derive(Clone, Copy, Debug)]
struct Model {
    weight: f64,
    bias: f64,
}

#[derive(Debug)]
struct Run {
    model: Model,
    train_curve: Vec<f64>,
    validation_curve: Vec<f64>,
}

impl Model {
    fn predict(self, x: f64) -> f64 {
        self.weight * x + self.bias
    }

    fn mse(self, data: &[(f64, f64)]) -> Result<f64, &'static str> {
        if data.is_empty() || data.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
            return Err("MSE requires finite examples");
        }
        let value = data
            .iter()
            .map(|&(x, y)| (self.predict(x) - y).powi(2))
            .sum::<f64>()
            / data.len() as f64;
        value.is_finite().then_some(value).ok_or("loss overflowed")
    }

    fn minibatch_step(
        self,
        batch: &[(f64, f64)],
        rate: f64,
        l2: f64,
    ) -> Result<Self, &'static str> {
        if batch.is_empty() || !rate.is_finite() || rate <= 0.0 || !l2.is_finite() || l2 < 0.0 {
            return Err("batch must be nonempty; rate positive; L2 nonnegative");
        }
        let n = batch.len() as f64;
        let (data_dw, db) = batch.iter().fold((0.0, 0.0), |(dw, db), &(x, y)| {
            let error = self.predict(x) - y;
            (dw + 2.0 * error * x / n, db + 2.0 * error / n)
        });
        let next = Self {
            weight: self.weight - rate * (data_dw + 2.0 * l2 * self.weight),
            bias: self.bias - rate * db,
        };
        if next.weight.is_finite() && next.bias.is_finite() {
            Ok(next)
        } else {
            Err("update diverged")
        }
    }
}

fn train(rate: f64, batch_size: usize, l2: f64, epochs: usize) -> Result<Run, &'static str> {
    if !rate.is_finite() || rate <= 0.0 || !l2.is_finite() || l2 < 0.0 {
        return Err("rate must be finite and positive; L2 finite and nonnegative");
    }
    if batch_size == 0 || batch_size > TRAIN.len() {
        return Err("batch size must be between one and the training length");
    }
    let mut model = Model {
        weight: 0.0,
        bias: 0.0,
    };
    let mut train_curve = Vec::with_capacity(epochs + 1);
    let mut validation_curve = Vec::with_capacity(epochs + 1);
    train_curve.push(model.mse(&TRAIN)?);
    validation_curve.push(model.mse(&VALIDATION)?);
    for _ in 0..epochs {
        // ponytail: fixed order keeps this trace reproducible; shuffle indices for stochastic training.
        for batch in TRAIN.chunks(batch_size) {
            model = model.minibatch_step(batch, rate, l2)?;
        }
        train_curve.push(model.mse(&TRAIN)?);
        validation_curve.push(model.mse(&VALIDATION)?);
    }
    Ok(Run {
        model,
        train_curve,
        validation_curve,
    })
}

fn report(label: &str, run: &Run) {
    let first = (run.train_curve[0], run.validation_curve[0]);
    let last = (
        *run.train_curve.last().unwrap_or(&f64::NAN),
        *run.validation_curve.last().unwrap_or(&f64::NAN),
    );
    println!(
        "{label:>12}: start train/val={:.3}/{:.3}, end={:.3}/{:.3}, w={:.3}, b={:.3}",
        first.0, first.1, last.0, last.1, run.model.weight, run.model.bias
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let slow = train(0.001, 4, 0.0, 80)?;
    let steady = train(0.03, 4, 0.0, 80)?;
    let regularized = train(0.03, 4, 0.1, 80)?;
    report("slow", &slow);
    report("steady", &steady);
    report("L2", &regularized);
    println!(
        "Curves are measured on fixed synthetic splits; compare validation, not training alone."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reliable_setting_improves_held_out_error() -> Result<(), &'static str> {
        let run = train(0.03, 4, 0.0, 80)?;
        assert!(
            run.validation_curve
                .last()
                .copied()
                .ok_or("missing curve")?
                < 0.2
        );
        assert!((run.model.weight - 2.0).abs() < 0.15);
        assert!(train(0.1, 0, 0.0, 1).is_err());
        Ok(())
    }

    #[test]
    fn l2_shrinks_the_weight_without_penalizing_bias() -> Result<(), &'static str> {
        let plain = train(0.03, 4, 0.0, 80)?;
        let regularized = train(0.03, 4, 0.2, 80)?;
        assert!(regularized.model.weight.abs() < plain.model.weight.abs());
        assert!(regularized
            .validation_curve
            .iter()
            .all(|loss| loss.is_finite()));
        Ok(())
    }
    #[test]
    fn short_batch_and_zero_epoch_validation() {
        let model = Model {
            weight: 2.0,
            bias: 1.0,
        };
        let next = model.minibatch_step(&[(0.0, 1.0)], 0.1, 0.5).unwrap();
        assert_eq!(next.weight, 1.8);
        assert_eq!(next.bias, 1.0);
        let data_only = model.minibatch_step(&[(1.0, 0.0)], 0.1, 0.0).unwrap();
        assert!((data_only.weight - 1.4).abs() < 1e-12);
        assert!((data_only.bias - 0.4).abs() < 1e-12);
        assert!(train(0.0, 4, 0.0, 0).is_err());
        assert!(train(0.1, 4, -1.0, 0).is_err());
        assert!(train(f64::NAN, 4, 0.0, 0).is_err());
    }
}
