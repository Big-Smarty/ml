//! A 2-2-1 network that learns XOR by manual backpropagation.

pub type Features = [f64; 2];
pub type Example = (Features, f64);

pub const XOR: [Example; 4] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];

pub fn sigmoid(logit: f64) -> f64 {
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exp = logit.exp();
        exp / (1.0 + exp)
    }
}

pub fn binary_cross_entropy_from_logit(logit: f64, target: f64) -> f64 {
    logit.max(0.0) - logit * target + (-logit.abs()).exp().ln_1p()
}

pub fn validate_data(data: &[Example]) -> Result<(), &'static str> {
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

pub fn validate_learning_rate(learning_rate: f64) -> Result<(), &'static str> {
    if learning_rate.is_finite() && learning_rate > 0.0 {
        Ok(())
    } else {
        Err("learning rate must be finite and positive")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Forward {
    pub hidden: [f64; 2],
    pub logit: f64,
    pub probability: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Gradient {
    pub hidden_weights: [[f64; 2]; 2],
    pub hidden_bias: [f64; 2],
    pub output_weights: [f64; 2],
    pub output_bias: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Net {
    pub hidden_weights: [[f64; 2]; 2],
    pub hidden_bias: [f64; 2],
    pub output_weights: [f64; 2],
    pub output_bias: f64,
}

impl Net {
    pub fn new() -> Self {
        Self {
            hidden_weights: [[0.7, -0.4], [-0.2, 0.9]],
            hidden_bias: [0.1, -0.3],
            output_weights: [0.8, -0.6],
            output_bias: 0.2,
        }
    }

    pub fn forward(&self, features: Features) -> Forward {
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

    pub fn probability(&self, features: Features) -> f64 {
        self.forward(features).probability
    }

    pub fn predict(&self, features: Features) -> u8 {
        u8::from(self.probability(features) >= 0.5)
    }

    pub fn loss(&self, data: &[Example]) -> Result<f64, &'static str> {
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

    pub fn step(
        mut self,
        data: &[Example],
        learning_rate: f64,
        derivative: Derivative,
    ) -> Result<Self, &'static str> {
        validate_learning_rate(learning_rate)?;
        let gradient = derivative(&self, data)?;
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

    pub fn train(
        mut self,
        data: &[Example],
        steps: usize,
        learning_rate: f64,
        derivative: Derivative,
    ) -> Result<Self, &'static str> {
        validate_learning_rate(learning_rate)?;
        for _ in 0..steps {
            self = self.step(data, learning_rate, derivative)?;
        }
        Ok(self)
    }
}

pub type Derivative = fn(&Net, &[Example]) -> Result<Gradient, &'static str>;
