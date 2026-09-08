//! A scalar neuron, trained with numerical slopes. No ML dependencies.
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
        let gradient = self.numerical_gradient(data)?;
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let initial = Neuron {
        weight: 0.0,
        bias: 0.0,
    };
    println!("initial training MSE: {:.6}", initial.loss(&TRAIN)?);
    let model = initial.train(&TRAIN, 100, 0.1)?;
    println!("weight: {:.6}, bias: {:.6}", model.weight, model.bias);
    println!("final training MSE: {:.10}", model.loss(&TRAIN)?);
    for input in [-1.5, 0.5, 3.0] {
        println!(
            "unseen input {input:.1} -> prediction {:.6}",
            model.predict(input)
        );
    }
    println!("These exact synthetic samples test the mechanism, not real-world generalization.");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hand_calculation_and_training() -> Result<(), &'static str> {
        assert_eq!(
            Neuron {
                weight: 0.0,
                bias: 0.0
            }
            .loss(&TRAIN)?,
            9.0
        );
        let step = Neuron {
            weight: 0.0,
            bias: 0.0,
        }
        .step(&TRAIN, 0.1)?;
        assert!((step.weight - 0.8).abs() < 1e-8 && (step.bias - 0.2).abs() < 1e-8);
        let gradient = Neuron {
            weight: 0.0,
            bias: 0.0,
        }
        .numerical_gradient(&TRAIN)?;
        assert!((gradient.weight + 8.0).abs() < 1e-8);
        assert!((gradient.bias + 2.0).abs() < 1e-8);
        let learned = Neuron {
            weight: 0.0,
            bias: 0.0,
        }
        .train(&TRAIN, 100, 0.1)?;
        assert!(learned.loss(&TRAIN)? < 1e-12);
        assert!((learned.predict(0.5) - 2.0).abs() < 1e-7);
        Ok(())
    }
    #[test]
    fn invalid_data_is_rejected() {
        let model = Neuron {
            weight: 0.0,
            bias: 0.0,
        };
        assert!(model.loss(&[]).is_err());
        assert!(model.loss(&[(f64::NAN, 1.0)]).is_err());
        assert!(model.step(&TRAIN, -1.0).is_err());
        assert!(model.train(&TRAIN, 0, f64::NAN).is_err());
        assert!(model.train(&[], 0, 0.1).is_err());
    }

    #[test]
    fn step_updates_both_parameters_from_the_same_model() -> Result<(), &'static str> {
        let data = [(1.0, 0.0), (2.0, 0.0)];
        let model = Neuron {
            weight: 1.0,
            bias: 1.0,
        }
        .step(&data, 0.1)?;
        assert!((model.weight - 0.2).abs() < 1e-8);
        assert!((model.bias - 0.5).abs() < 1e-8);
        Ok(())
    }
}
