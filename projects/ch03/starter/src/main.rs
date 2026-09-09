type Features = [f64; 3];
type Example = (Features, f64);

const DATA: [Example; 2] = [([2.0, 3.0, 1.0], 7.0), ([-1.0, 2.0, 0.0], 1.0)];

#[derive(Clone, Copy, Debug)]
struct LinearModel {
    weights: [f64; 3],
    bias: f64,
}

#[derive(Clone, Copy, Debug)]
struct Gradient {
    weights: [f64; 3],
    bias: f64,
}

impl LinearModel {
    fn predict(&self, features: Features) -> f64 {
        self.weights
            .iter()
            .zip(features)
            .map(|(weight, feature)| weight * feature)
            .sum::<f64>()
            + self.bias
    }

    fn loss(&self, data: &[Example]) -> Result<f64, &'static str> {
        if data.is_empty() {
            return Err("training requires examples");
        }
        let loss = data
            .iter()
            .map(|&(features, target)| (self.predict(features) - target).powi(2))
            .sum::<f64>()
            / data.len() as f64;
        loss.is_finite().then_some(loss).ok_or("loss overflowed")
    }

    fn loss_and_gradient(&self, data: &[Example]) -> Result<(f64, Gradient), &'static str> {
        // TODO: extend Chapter 2's scalar accumulation across all three features.
        let n = data.len() as f64;
        let loss = self.loss(data)?;
        let weights_gradients = [
            2.0 / n
                * (0..data.len())
                    .map(|i| {
                        (self
                            .weights
                            .iter()
                            .zip(data[i].0)
                            .map(|(w, x)| w * x)
                            .sum::<f64>()
                            + self.bias
                            - data[i].1)
                            * data[i].0[0]
                    })
                    .sum::<f64>(),
            2.0 / n
                * (0..data.len())
                    .map(|i| {
                        (self
                            .weights
                            .iter()
                            .zip(data[i].0)
                            .map(|(w, x)| w * x)
                            .sum::<f64>()
                            + self.bias
                            - data[i].1)
                            * data[i].0[1]
                    })
                    .sum::<f64>(),
            2.0 / n
                * (0..data.len())
                    .map(|i| {
                        (self
                            .weights
                            .iter()
                            .zip(data[i].0)
                            .map(|(w, x)| w * x)
                            .sum::<f64>()
                            + self.bias
                            - data[i].1)
                            * data[i].0[2]
                    })
                    .sum::<f64>(),
        ];
        let bias_gradient = data.iter().map(|d| self.predict(d.0) - d.1).sum::<f64>();
        Ok((
            loss,
            Gradient {
                weights: weights_gradients,
                bias: bias_gradient,
            },
        ))
    }

    fn gradient(&self, data: &[Example]) -> Result<Gradient, &'static str> {
        self.loss_and_gradient(data).map(|(_, gradient)| gradient)
    }

    fn step(self, data: &[Example], learning_rate: f64) -> Result<Self, &'static str> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        let gradient = self.gradient(data)?;
        let mut next = self;
        for (weight, weight_gradient) in next.weights.iter_mut().zip(gradient.weights) {
            *weight -= learning_rate * weight_gradient;
        }
        next.bias -= learning_rate * gradient.bias;
        next.loss(data)?;
        Ok(next)
    }

    fn train(
        mut self,
        data: &[Example],
        steps: usize,
        learning_rate: f64,
    ) -> Result<Self, &'static str> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        for _ in 0..steps {
            self = self.step(data, learning_rate)?;
        }
        Ok(self)
    }
}

fn main() -> Result<(), &'static str> {
    let model = LinearModel {
        weights: [0.0; 3],
        bias: 0.0,
    };
    let _training_checkpoint: fn(
        LinearModel,
        &[Example],
        usize,
        f64,
    ) -> Result<LinearModel, &'static str> = LinearModel::train;
    println!(
        "three-feature prediction: {}",
        model.predict([2.0, 3.0, 1.0])
    );
    println!("initial MSE: {}", model.loss(&DATA)?);
    println!("Run cargo test to implement loss_and_gradient.");
    Ok(())
}

#[test]
fn one_step_uses_every_feature() -> Result<(), &'static str> {
    let model = LinearModel {
        weights: [0.0; 3],
        bias: 0.0,
    };
    let gradient = model.gradient(&DATA)?;
    assert_eq!(gradient.weights, [-13.0, -23.0, -7.0]);
    assert_eq!(gradient.bias, -8.0);
    let next = model.step(&DATA, 0.01)?;
    assert!(next.loss(&DATA)? < model.loss(&DATA)?);
    Ok(())
}
