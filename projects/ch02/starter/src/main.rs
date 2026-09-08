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
            return Err("at least one example is required");
        }
        let loss = data
            .iter()
            .map(|&(input, target)| (self.predict(input) - target).powi(2))
            .sum::<f64>()
            / data.len() as f64;
        loss.is_finite().then_some(loss).ok_or("loss overflowed")
    }

    fn gradient(&self, data: &[(f64, f64)]) -> Result<Gradient, &'static str> {
        // TODO: average 2 * error * input and 2 * error over data.
        let _ = data;
        todo!("return the analytical weight and bias gradient")
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
        if weight.is_finite() && bias.is_finite() {
            Ok(Gradient { weight, bias })
        } else {
            Err("numerical gradient overflowed")
        }
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
        mut self,
        data: &[(f64, f64)],
        steps: usize,
        learning_rate: f64,
    ) -> Result<Self, &'static str> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        self.loss(data)?;
        for _ in 0..steps {
            self = self.step(data, learning_rate)?;
        }
        Ok(self)
    }
}

fn main() -> Result<(), &'static str> {
    let _guided_gradient = Neuron::gradient;
    let _guided_step = Neuron::step;
    let _guided_train = Neuron::train;
    let model = Neuron {
        weight: 0.0,
        bias: 0.0,
    };
    let numerical = model.numerical_gradient(&TRAIN)?;
    println!("initial loss: {:.1}", model.loss(&TRAIN)?);
    println!(
        "Chapter 1 numerical gradient: weight={:.6}, bias={:.6}",
        numerical.weight, numerical.bias
    );
    println!("Run cargo test to implement the analytical-gradient TODO.");
    Ok(())
}

#[test]
fn analytical_gradient_drives_the_same_update() -> Result<(), &'static str> {
    let model = Neuron {
        weight: 0.0,
        bias: 0.0,
    };
    let gradient = model.gradient(&TRAIN)?;
    assert!((gradient.weight + 8.0).abs() < 1e-12);
    assert!((gradient.bias + 2.0).abs() < 1e-12);
    let next = model.step(&TRAIN, 0.1)?;
    assert!((next.weight - 0.8).abs() < 1e-12);
    assert!((next.bias - 0.2).abs() < 1e-12);
    let trained = model.train(&TRAIN, 100, 0.1)?;
    assert!(trained.loss(&TRAIN)? < 1e-12);
    Ok(())
}
