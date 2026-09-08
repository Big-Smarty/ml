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
                let logit = self.forward(features).logit;
                logit.max(0.0) - logit * target + (-logit.abs()).exp().ln_1p()
            })
            .sum::<f64>()
            / data.len() as f64;
        loss.is_finite().then_some(loss).ok_or("loss overflowed")
    }

    fn gradient(&self, data: &[Example]) -> Result<Gradient, &'static str> {
        validate_data(data)?;
        // TODO: average every parameter derivative over the supplied examples.
        let _ = data;
        todo!("implement full-batch backpropagation")
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Net::new();
    let _guided = Net::gradient;
    let gradient_shape = Gradient::default();
    let gradient_slots = gradient_shape.hidden_weights.iter().flatten().count()
        + gradient_shape.hidden_bias.len()
        + gradient_shape.output_weights.len()
        + usize::from(gradient_shape.output_bias == 0.0);
    let forward = model.forward([1.0, 0.0]);
    println!(
        "hidden={:?}, logit={:.4}, probability={:.4}, class={}",
        forward.hidden,
        forward.logit,
        forward.probability,
        model.predict([1.0, 0.0])
    );
    println!("initial mean BCE: {:.6}", model.loss(&XOR)?);
    println!("the Gradient has {gradient_slots} parameter derivatives");
    Ok(())
}

#[test]
fn gradient_contains_mean_parameter_derivatives() {
    let gradient = Net::new().gradient(&XOR).unwrap();
    assert!((gradient.hidden_weights[1][0] + 0.007_777_711_854).abs() < 1e-9);
    assert!((gradient.hidden_bias[0] - 0.018_793_782_868).abs() < 1e-9);
    assert!((gradient.output_weights[0] - 0.050_928_541_826).abs() < 1e-9);
    assert!((gradient.output_bias - 0.084_009_156_432).abs() < 1e-9);
}
