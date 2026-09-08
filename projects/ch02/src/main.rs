//! The Chapter 1 scalar neuron, now trained with an analytical gradient.

const TRAIN: [(f64, f64); 5] = [
    (-2.0, -3.0),
    (-1.0, -1.0),
    (0.0, 1.0),
    (1.0, 3.0),
    (2.0, 5.0),
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
        if weight.is_finite() && bias.is_finite() {
            Ok(Gradient { weight, bias })
        } else {
            Err("gradient overflowed")
        }
    }

    fn numerical_gradient(&self, data: &[(f64, f64)]) -> Result<Gradient, &'static str> {
        const H: f64 = 1e-5;
        let weight = (Self {
            weight: self.weight + H,
            ..*self
        }
        .loss(data)?
            - Self {
                weight: self.weight - H,
                ..*self
            }
            .loss(data)?)
            / (2.0 * H);
        let bias = (Self {
            bias: self.bias + H,
            ..*self
        }
        .loss(data)?
            - Self {
                bias: self.bias - H,
                ..*self
            }
            .loss(data)?)
            / (2.0 * H);
        if !weight.is_finite() || !bias.is_finite() {
            return Err("numerical gradient overflowed; reduce input scale");
        }
        Ok(Gradient { weight, bias })
    }

    fn step(self, data: &[(f64, f64)], learning_rate: f64) -> Result<Self, &'static str> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        let gradient = self.gradient(data)?;
        let next = Self {
            weight: self.weight - learning_rate * gradient.weight,
            bias: self.bias - learning_rate * gradient.bias,
        };
        next.loss(data)?;
        Ok(next)
    }

    fn train(
        self,
        data: &[(f64, f64)],
        steps: usize,
        learning_rate: f64,
    ) -> Result<Self, &'static str> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        self.loss(data)?;
        let mut model = self;
        for _ in 0..steps {
            model = model.step(data, learning_rate)?;
        }
        Ok(model)
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 + 1e-4 * a.abs().max(b.abs())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let initial = Neuron {
        weight: 0.0,
        bias: 0.0,
    };
    let analytical = initial.gradient(&TRAIN)?;
    let numerical = initial.numerical_gradient(&TRAIN)?;
    println!(
        "analytical weight={:.6}, bias={:.6}",
        analytical.weight, analytical.bias
    );
    println!(
        "numerical  weight={:.6}, bias={:.6}",
        numerical.weight, numerical.bias
    );
    println!(
        "check: {}",
        close(analytical.weight, numerical.weight) && close(analytical.bias, numerical.bias)
    );
    let model = initial.train(&TRAIN, 100, 0.1)?;
    println!(
        "trained weight={:.6}, bias={:.6}, loss={:.10}",
        model.weight,
        model.bias,
        model.loss(&TRAIN)?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytical_gradient_matches_central_difference() -> Result<(), &'static str> {
        for model in [
            Neuron {
                weight: 0.0,
                bias: 0.0,
            },
            Neuron {
                weight: 1.3,
                bias: -0.4,
            },
        ] {
            let analytical = model.gradient(&TRAIN)?;
            let numerical = model.numerical_gradient(&TRAIN)?;
            assert!(close(analytical.weight, numerical.weight));
            assert!(close(analytical.bias, numerical.bias));
        }
        Ok(())
    }

    #[test]
    fn one_step_and_training_use_the_expected_derivative() -> Result<(), &'static str> {
        let initial = Neuron {
            weight: 0.0,
            bias: 0.0,
        };
        assert!(close(initial.loss(&TRAIN)?, 9.0));
        let gradient = initial.gradient(&TRAIN)?;
        assert!(close(gradient.weight, -8.0));
        assert!(close(gradient.bias, -2.0));
        let next = initial.step(&TRAIN, 0.1)?;
        assert!(close(next.weight, 0.8));
        assert!(close(next.bias, 0.2));
        let model = initial.train(&TRAIN, 100, 0.1)?;
        assert!(model.loss(&TRAIN)? < 1e-12);
        assert!(initial.loss(&[]).is_err());
        assert!(initial.gradient(&[(f64::NAN, 1.0)]).is_err());
        assert!(initial.train(&[], 0, 0.1).is_err());
        assert!(initial.train(&TRAIN, 0, f64::NAN).is_err());
        assert!(Neuron {
            weight: 0.1,
            bias: 0.0,
        }
        .numerical_gradient(&[(1e155, 0.0)])
        .is_err());
        assert!(initial.step(&TRAIN, -1.0).is_err());
        Ok(())
    }
}
