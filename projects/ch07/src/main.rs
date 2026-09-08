//! A 2-2-1 network that learns XOR by manual backpropagation.

type Features = [f64; 2];
type Example = (Features, f64);

const XOR: [Example; 4] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];

fn sigmoid(logit: f64) -> f64 {
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exp = logit.exp();
        exp / (1.0 + exp)
    }
}

fn binary_cross_entropy_from_logit(logit: f64, target: f64) -> f64 {
    logit.max(0.0) - logit * target + (-logit.abs()).exp().ln_1p()
}

fn validate_data(data: &[Example]) -> Result<(), &'static str> {
    if data.is_empty() {
        return Err("training requires examples");
    }
    if data.iter().any(|(features, target)| {
        features.iter().any(|feature| !feature.is_finite())
            || !target.is_finite()
            || !(0.0..=1.0).contains(target)
    }) {
        return Err("features must be finite and targets must be in [0, 1]");
    }
    Ok(())
}

fn validate_learning_rate(learning_rate: f64) -> Result<(), &'static str> {
    if learning_rate.is_finite() && learning_rate > 0.0 {
        Ok(())
    } else {
        Err("learning rate must be finite and positive")
    }
}

#[derive(Clone, Copy, Debug)]
struct Forward {
    hidden: [f64; 2],
    logit: f64,
    probability: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Gradient {
    hidden_weights: [[f64; 2]; 2],
    hidden_bias: [f64; 2],
    output_weights: [f64; 2],
    output_bias: f64,
}

#[derive(Clone, Copy, Debug)]
struct Net {
    hidden_weights: [[f64; 2]; 2],
    hidden_bias: [f64; 2],
    output_weights: [f64; 2],
    output_bias: f64,
}

impl Net {
    fn new() -> Self {
        Self {
            hidden_weights: [[0.7, -0.4], [-0.2, 0.9]],
            hidden_bias: [0.1, -0.3],
            output_weights: [0.8, -0.6],
            output_bias: 0.2,
        }
    }

    fn forward(&self, features: Features) -> Forward {
        let hidden = [
            sigmoid(
                self.hidden_weights[0][0] * features[0]
                    + self.hidden_weights[0][1] * features[1]
                    + self.hidden_bias[0],
            ),
            sigmoid(
                self.hidden_weights[1][0] * features[0]
                    + self.hidden_weights[1][1] * features[1]
                    + self.hidden_bias[1],
            ),
        ];
        let logit = self.output_weights[0] * hidden[0]
            + self.output_weights[1] * hidden[1]
            + self.output_bias;
        Forward {
            hidden,
            logit,
            probability: sigmoid(logit),
        }
    }

    fn probability(&self, features: Features) -> f64 {
        self.forward(features).probability
    }

    fn predict(&self, features: Features) -> u8 {
        u8::from(self.probability(features) >= 0.5)
    }

    fn loss(&self, data: &[Example]) -> Result<f64, &'static str> {
        validate_data(data)?;
        let loss = data
            .iter()
            .map(|&(features, target)| {
                binary_cross_entropy_from_logit(self.forward(features).logit, target)
            })
            .sum::<f64>()
            / data.len() as f64;
        loss.is_finite().then_some(loss).ok_or("loss overflowed")
    }

    fn gradient(&self, data: &[Example]) -> Result<Gradient, &'static str> {
        validate_data(data)?;
        let n = data.len() as f64;
        let mut gradient = Gradient::default();
        for &(features, target) in data {
            let forward = self.forward(features);
            let output_signal = forward.probability - target;
            for j in 0..2 {
                gradient.output_weights[j] += output_signal * forward.hidden[j] / n;
                let hidden_signal = output_signal
                    * self.output_weights[j]
                    * forward.hidden[j]
                    * (1.0 - forward.hidden[j]);
                gradient.hidden_bias[j] += hidden_signal / n;
                for (weight_gradient, feature) in
                    gradient.hidden_weights[j].iter_mut().zip(features)
                {
                    *weight_gradient += hidden_signal * feature / n;
                }
            }
            gradient.output_bias += output_signal / n;
        }
        Ok(gradient)
    }

    fn step(mut self, data: &[Example], learning_rate: f64) -> Result<Self, &'static str> {
        validate_learning_rate(learning_rate)?;
        let gradient = self.gradient(data)?;
        for j in 0..2 {
            for k in 0..2 {
                self.hidden_weights[j][k] -= learning_rate * gradient.hidden_weights[j][k];
            }
            self.hidden_bias[j] -= learning_rate * gradient.hidden_bias[j];
            self.output_weights[j] -= learning_rate * gradient.output_weights[j];
        }
        self.output_bias -= learning_rate * gradient.output_bias;
        self.loss(data)?;
        Ok(self)
    }

    fn train(
        mut self,
        data: &[Example],
        steps: usize,
        learning_rate: f64,
    ) -> Result<Self, &'static str> {
        validate_learning_rate(learning_rate)?;
        for _ in 0..steps {
            self = self.step(data, learning_rate)?;
        }
        Ok(self)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Net::new().train(&XOR, 10_000, 1.0)?;
    println!("mean binary cross-entropy: {:.6}", model.loss(&XOR)?);
    for &(features, target) in &XOR {
        println!(
            "{features:?} target {target:.0} probability {:.4} class {}",
            model.probability(features),
            model.predict(features)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameter(net: &mut Net, index: usize) -> &mut f64 {
        match index {
            0..=3 => &mut net.hidden_weights[index / 2][index % 2],
            4..=5 => &mut net.hidden_bias[index - 4],
            6..=7 => &mut net.output_weights[index - 6],
            8 => &mut net.output_bias,
            _ => unreachable!(),
        }
    }

    fn gradient_parameter(gradient: Gradient, index: usize) -> f64 {
        match index {
            0..=3 => gradient.hidden_weights[index / 2][index % 2],
            4..=5 => gradient.hidden_bias[index - 4],
            6..=7 => gradient.output_weights[index - 6],
            8 => gradient.output_bias,
            _ => unreachable!(),
        }
    }

    #[test]
    fn learns_xor() -> Result<(), &'static str> {
        let model = Net::new().train(&XOR, 10_000, 1.0)?;
        assert!(model.loss(&XOR)? < 0.02);
        for &(features, target) in &XOR {
            assert_eq!(model.predict(features), target as u8);
        }
        Ok(())
    }

    #[test]
    fn gradient_matches_central_differences() -> Result<(), &'static str> {
        let model = Net::new();
        let gradient = model.gradient(&XOR)?;
        for index in 0..9 {
            let mut plus = model;
            let mut minus = model;
            *parameter(&mut plus, index) += 1e-5;
            *parameter(&mut minus, index) -= 1e-5;
            let numerical = (plus.loss(&XOR)? - minus.loss(&XOR)?) / 2e-5;
            let analytical = gradient_parameter(gradient, index);
            assert!((analytical - numerical).abs() < 1e-6 + 1e-4 * numerical.abs());
        }
        Ok(())
    }

    #[test]
    fn step_applies_the_mean_gradient() -> Result<(), &'static str> {
        let model = Net::new();
        let gradient = model.gradient(&XOR)?;
        let updated = model.step(&XOR, 0.1)?;
        for index in 0..9 {
            let mut before = model;
            let mut after = updated;
            let actual = (*parameter(&mut before, index) - *parameter(&mut after, index)) / 0.1;
            assert!((actual - gradient_parameter(gradient, index)).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn invalid_data_and_learning_rates_are_rejected() {
        assert!(Net::new().loss(&[]).is_err());
        assert!(Net::new().loss(&[([f64::NAN, 0.0], 0.0)]).is_err());
        assert!(Net::new().loss(&[([0.0, 0.0], 2.0)]).is_err());
        assert!(Net::new().step(&XOR, 0.0).is_err());
    }
}
