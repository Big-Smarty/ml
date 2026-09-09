//! Supplied scalar oracle, identical 17-parameter layout to the WGSL trainer.
pub type Features = [f32; 2];
pub type Example = (Features, f32);

pub const TRAIN_DATA: [Example; 8] = [
    ([-1.0, -1.0], 0.0),
    ([-1.0, 1.0], 1.0),
    ([1.0, -1.0], 1.0),
    ([1.0, 1.0], 0.0),
    ([-0.8, -1.2], 0.0),
    ([-1.2, 0.8], 1.0),
    ([0.8, -1.2], 1.0),
    ([1.2, 0.8], 0.0),
];

#[derive(Clone, Debug)]
pub struct Model {
    // 0..8: hidden input weights; 8..12: hidden biases;
    // 12..16: output weights; 16: output bias.
    pub parameters: [f32; 17],
}

pub const INITIAL: Model = Model {
    parameters: [
        0.30, -0.20, -0.40, 0.35, 0.25, 0.45, -0.35, -0.25, 0.05, -0.05, 0.10, -0.10, 0.40, -0.30,
        0.25, -0.35, 0.0,
    ],
};

#[derive(Debug)]
pub struct Gradient {
    pub parameters: [f32; 17],
}

#[derive(Debug)]
pub struct TrainingResult {
    pub model: Model,
    pub losses: Vec<f32>,
}

impl Model {
    pub fn forward(&self, features: Features) -> ([f32; 4], f32) {
        let hidden = std::array::from_fn(|unit| {
            (self.parameters[unit * 2] * features[0]
                + self.parameters[unit * 2 + 1] * features[1]
                + self.parameters[8 + unit])
                .tanh()
        });
        let logit = self.parameters[16]
            + (0..4)
                .map(|unit| self.parameters[12 + unit] * hidden[unit])
                .sum::<f32>();
        (hidden, logit)
    }

    pub fn probability(&self, features: Features) -> f32 {
        sigmoid(self.forward(features).1)
    }

    pub fn loss(&self, data: &[Example]) -> Result<f32, &'static str> {
        validate_data(data)?;
        let mut loss = 0.0;
        for &(features, target) in data {
            loss += binary_cross_entropy_from_logit(self.forward(features).1, target)?;
        }
        Ok(loss / data.len() as f32)
    }

    pub fn gradient(&self, data: &[Example]) -> Result<Gradient, &'static str> {
        validate_data(data)?;
        let mut gradient = Gradient {
            parameters: [0.0; 17],
        };
        for &(features, target) in data {
            let (hidden, logit) = self.forward(features);
            let output_delta = sigmoid(logit) - target;
            for (unit, &activation) in hidden.iter().enumerate() {
                gradient.parameters[12 + unit] += output_delta * activation;
                let hidden_delta =
                    output_delta * self.parameters[12 + unit] * (1.0 - activation * activation);
                gradient.parameters[unit * 2] += hidden_delta * features[0];
                gradient.parameters[unit * 2 + 1] += hidden_delta * features[1];
                gradient.parameters[8 + unit] += hidden_delta;
            }
            gradient.parameters[16] += output_delta;
        }
        for value in &mut gradient.parameters {
            *value /= data.len() as f32;
        }
        Ok(gradient)
    }

    pub fn step(&mut self, data: &[Example], learning_rate: f32) -> Result<(), &'static str> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning_rate must be finite and positive");
        }
        let gradient = self.gradient(data)?;
        for (parameter, slope) in self.parameters.iter_mut().zip(gradient.parameters) {
            *parameter -= learning_rate * slope;
        }
        Ok(())
    }

    pub fn train(
        mut self,
        data: &[Example],
        steps: usize,
        learning_rate: f32,
    ) -> Result<Self, &'static str> {
        validate_data(data)?;
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning_rate must be finite and positive");
        }
        for _ in 0..steps {
            self.step(data, learning_rate)?;
        }
        Ok(self)
    }
}

fn sigmoid(logit: f32) -> f32 {
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exp = logit.exp();
        exp / (1.0 + exp)
    }
}

pub fn validate_data(data: &[Example]) -> Result<(), &'static str> {
    if data.is_empty() {
        return Err("training requires examples");
    }
    if data.iter().any(|(features, target)| {
        features.iter().any(|value| !value.is_finite())
            || !target.is_finite()
            || !(0.0..=1.0).contains(target)
    }) {
        return Err("features must be finite and targets must be in [0, 1]");
    }
    Ok(())
}

pub fn binary_cross_entropy_from_logit(logit: f32, target: f32) -> Result<f32, &'static str> {
    if !logit.is_finite() || !target.is_finite() || !(0.0..=1.0).contains(&target) {
        return Err("logit must be finite and target must be in [0, 1]");
    }
    Ok(logit.max(0.0) - logit * target + (-logit.abs()).exp().ln_1p())
}
