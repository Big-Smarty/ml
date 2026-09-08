// The learner completed prediction and extended this starter through numerical training.
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
    for (input, target) in TRAIN {
        println!("input={input:>4}, target={target:>4}");
    }
    let initial = Neuron {
        weight: 0.0,
        bias: 0.0,
    };
    println!("initial loss: {}", initial.loss(&TRAIN)?);
    let model = initial.train(&TRAIN, 1000, 0.1)?;
    println!("weight: {}, bias: {}", model.weight, model.bias);
    println!("final loss: {}", model.loss(&TRAIN)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_prediction_and_training() -> Result<(), &'static str> {
        let initial = Neuron {
            weight: 0.0,
            bias: 0.0,
        };
        assert!((initial.predict(0.5) - 0.0).abs() < 1e-12);
        assert_eq!(
            Neuron {
                weight: -1.0,
                bias: 4.0,
            }
            .predict(3.0),
            1.0
        );
        let model = initial.train(&TRAIN, 1000, 0.1)?;
        assert!((model.predict(0.5) - 2.0).abs() < 1e-12);
        assert!((model.weight - 2.0).abs() < 1e-12);
        assert!((model.bias - 1.0).abs() < 1e-12);
        Ok(())
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

    #[test]
    fn invalid_zero_step_training_is_rejected() {
        let model = Neuron {
            weight: 0.0,
            bias: 0.0,
        };
        assert!(model.train(&[], 0, 0.1).is_err());
        assert!(model.train(&TRAIN, 0, f64::NAN).is_err());
    }
}
