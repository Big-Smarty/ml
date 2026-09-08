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
struct Neuron {
    weight: f64,
    bias: f64,
}

#[derive(Clone, Copy, Debug)]
struct Gradient {
    weight: f64,
    bias: f64,
}

#[derive(Debug)]
struct Run {
    model: Neuron,
    train_curve: Vec<f64>,
    validation_curve: Vec<f64>,
}

fn validate(data: &[(f64, f64)]) -> Result<(), &'static str> {
    if data.is_empty() {
        return Err("at least one example is required");
    }
    if data
        .iter()
        .any(|(input, target)| !input.is_finite() || !target.is_finite())
    {
        return Err("examples must be finite");
    }
    Ok(())
}

impl Neuron {
    fn predict(&self, input: f64) -> f64 {
        self.weight * input + self.bias
    }

    fn loss(&self, data: &[(f64, f64)]) -> Result<f64, &'static str> {
        if data.is_empty() {
            return Err("loss requires at least one example");
        }
        if !self.weight.is_finite()
            || !self.bias.is_finite()
            || data
                .iter()
                .any(|(input, target)| !input.is_finite() || !target.is_finite())
        {
            return Err("model and examples must contain finite numbers");
        }
        let loss = data
            .iter()
            .map(|&(input, target)| (self.predict(input) - target).powi(2))
            .sum::<f64>()
            / data.len() as f64;
        if !loss.is_finite() {
            return Err("loss overflowed; reduce input scale or learning rate");
        }
        Ok(loss)
    }

    fn gradient(&self, data: &[(f64, f64)]) -> Result<Gradient, &'static str> {
        validate(data)?;
        let n = data.len() as f64;
        let (weight, bias) = data
            .iter()
            .fold((0.0, 0.0), |(weight, bias), &(input, target)| {
                let error = self.predict(input) - target;
                (weight + 2.0 * error * input / n, bias + 2.0 * error / n)
            });
        if !weight.is_finite() || !bias.is_finite() {
            return Err("gradient overflowed; reduce input scale");
        }
        Ok(Gradient { weight, bias })
    }

    fn step(self, batch: &[(f64, f64)], learning_rate: f64, l2: f64) -> Result<Self, &'static str> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        if !l2.is_finite() || l2 < 0.0 {
            return Err("L2 strength must be finite and nonnegative");
        }
        let data_gradient = self.gradient(batch)?;
        let gradient = Gradient {
            weight: data_gradient.weight + 2.0 * l2 * self.weight,
            bias: data_gradient.bias,
        };
        let next = Self {
            weight: self.weight - learning_rate * gradient.weight,
            bias: self.bias - learning_rate * gradient.bias,
        };
        next.loss(batch)?;
        Ok(next)
    }

    fn train(
        self,
        train_data: &[(f64, f64)],
        validation_data: &[(f64, f64)],
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        l2: f64,
    ) -> Result<Run, &'static str> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        if !l2.is_finite() || l2 < 0.0 {
            return Err("L2 strength must be finite and nonnegative");
        }
        let initial_train_loss = self.loss(train_data)?;
        let initial_validation_loss = self.loss(validation_data)?;
        if batch_size == 0 || batch_size > train_data.len() {
            return Err("batch size must be between one and the training length");
        }
        let mut model = self;
        let mut train_curve = Vec::with_capacity(epochs + 1);
        let mut validation_curve = Vec::with_capacity(epochs + 1);
        train_curve.push(initial_train_loss);
        validation_curve.push(initial_validation_loss);
        for _ in 0..epochs {
            // ponytail: fixed order keeps this trace reproducible; shuffle indices for stochastic training.
            for batch in train_data.chunks(batch_size) {
                model = model.step(batch, learning_rate, l2)?;
            }
            train_curve.push(model.loss(train_data)?);
            validation_curve.push(model.loss(validation_data)?);
        }
        Ok(Run {
            model,
            train_curve,
            validation_curve,
        })
    }
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
    let model = Neuron {
        weight: 0.0,
        bias: 0.0,
    };
    let slow = model.train(&TRAIN, &VALIDATION, 80, 4, 0.001, 0.0)?;
    let steady = model.train(&TRAIN, &VALIDATION, 80, 4, 0.03, 0.0)?;
    let regularized = model.train(&TRAIN, &VALIDATION, 80, 4, 0.03, 0.1)?;
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
        let model = Neuron {
            weight: 0.0,
            bias: 0.0,
        };
        let run = model.train(&TRAIN, &VALIDATION, 80, 4, 0.03, 0.0)?;
        assert!(
            run.validation_curve
                .last()
                .copied()
                .ok_or("missing curve")?
                < 0.2
        );
        assert!((run.model.weight - 2.0).abs() < 0.15);
        assert!(model.train(&TRAIN, &VALIDATION, 1, 0, 0.1, 0.0).is_err());
        Ok(())
    }

    #[test]
    fn l2_shrinks_the_weight_without_penalizing_bias() -> Result<(), &'static str> {
        let model = Neuron {
            weight: 0.0,
            bias: 0.0,
        };
        let plain = model.train(&TRAIN, &VALIDATION, 80, 4, 0.03, 0.0)?;
        let regularized = model.train(&TRAIN, &VALIDATION, 80, 4, 0.03, 0.2)?;
        assert!(regularized.model.weight.abs() < plain.model.weight.abs());
        assert!(regularized
            .validation_curve
            .iter()
            .all(|loss| loss.is_finite()));
        Ok(())
    }
    #[test]
    fn short_batch_and_zero_epoch_validation() -> Result<(), &'static str> {
        let model = Neuron {
            weight: 2.0,
            bias: 1.0,
        };
        let next = model.step(&[(0.0, 1.0)], 0.1, 0.5)?;
        assert_eq!(next.weight, 1.8);
        assert_eq!(next.bias, 1.0);
        let data_only = model.step(&[(1.0, 0.0)], 0.1, 0.0)?;
        assert!((data_only.weight - 1.4).abs() < 1e-12);
        assert!((data_only.bias - 0.4).abs() < 1e-12);
        assert!(model.train(&TRAIN, &VALIDATION, 0, 4, 0.0, 0.0).is_err());
        assert!(model.train(&TRAIN, &VALIDATION, 0, 4, 0.1, -1.0).is_err());
        assert!(model
            .train(&TRAIN, &VALIDATION, 0, 4, f64::NAN, 0.0)
            .is_err());
        Ok(())
    }
}
