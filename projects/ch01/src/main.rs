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
impl Neuron {
    fn predict(self, input: f64) -> f64 {
        self.weight * input + self.bias
    }
    fn loss(self, data: &[(f64, f64)]) -> Result<f64, &'static str> {
        if data.is_empty() {
            return Err("loss requires at least one example");
        }
        if !self.weight.is_finite()
            || !self.bias.is_finite()
            || data.iter().any(|(x, y)| !x.is_finite() || !y.is_finite())
        {
            return Err("model and examples must contain finite numbers");
        }
        let loss = data
            .iter()
            .map(|&(x, y)| (self.predict(x) - y).powi(2))
            .sum::<f64>()
            / data.len() as f64;
        if !loss.is_finite() {
            return Err("loss overflowed; reduce input scale or learning rate");
        }
        Ok(loss)
    }
    fn step(self, data: &[(f64, f64)], rate: f64) -> Result<Self, &'static str> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        let h = 1e-5;
        let dw = (Self {
            weight: self.weight + h,
            ..self
        }
        .loss(data)?
            - Self {
                weight: self.weight - h,
                ..self
            }
            .loss(data)?)
            / (2.0 * h);
        let db = (Self {
            bias: self.bias + h,
            ..self
        }
        .loss(data)?
            - Self {
                bias: self.bias - h,
                ..self
            }
            .loss(data)?)
            / (2.0 * h);
        let next = Self {
            weight: self.weight - rate * dw,
            bias: self.bias - rate * db,
        };
        next.loss(data)?;
        Ok(next)
    }
}
fn train(steps: usize, rate: f64) -> Result<Neuron, &'static str> {
    let mut model = Neuron {
        weight: 0.0,
        bias: 0.0,
    };
    for _ in 0..steps {
        model = model.step(&TRAIN, rate)?;
    }
    Ok(model)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let initial = Neuron {
        weight: 0.0,
        bias: 0.0,
    };
    println!("initial training MSE: {:.6}", initial.loss(&TRAIN)?);
    let model = train(100, 0.1)?;
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
        let learned = train(100, 0.1)?;
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
    }
}
