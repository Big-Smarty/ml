//! Binary logistic regression trained directly from stable logits.

type Features = [f64; 2];
type Example = (Features, f64);

const TRAIN_DATA: [Example; 8] = [
    ([-2.0, -1.0], 0.0),
    ([-1.5, 0.2], 0.0),
    ([-0.8, -1.3], 0.0),
    ([-0.2, -0.7], 0.0),
    ([0.4, 0.8], 1.0),
    ([0.9, 1.5], 1.0),
    ([1.4, 0.1], 1.0),
    ([2.0, 1.0], 1.0),
];

#[derive(Clone, Copy, Debug)]
struct LogisticModel {
    weights: [f64; 2],
    bias: f64,
}

#[derive(Clone, Copy, Debug)]
struct Gradient {
    weights: [f64; 2],
    bias: f64,
}

fn sigmoid(logit: f64) -> f64 {
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exp = logit.exp();
        exp / (1.0 + exp)
    }
}

fn binary_cross_entropy_from_logit(logit: f64, target: f64) -> Result<f64, &'static str> {
    if !logit.is_finite() || !target.is_finite() || !(0.0..=1.0).contains(&target) {
        return Err("logit must be finite and target must be in [0, 1]");
    }
    Ok(logit.max(0.0) - logit * target + (-logit.abs()).exp().ln_1p())
}

fn validate_learning_rate(learning_rate: f64) -> Result<(), &'static str> {
    if learning_rate.is_finite() && learning_rate > 0.0 {
        Ok(())
    } else {
        Err("learning rate must be finite and positive")
    }
}

impl LogisticModel {
    fn logit(&self, features: Features) -> f64 {
        self.weights[0] * features[0] + self.weights[1] * features[1] + self.bias
    }

    fn probability(&self, features: Features) -> f64 {
        sigmoid(self.logit(features))
    }

    fn predict(&self, features: Features) -> u8 {
        u8::from(self.probability(features) >= 0.5)
    }

    fn loss(&self, data: &[Example]) -> Result<f64, &'static str> {
        self.loss_and_gradient(data).map(|(loss, _)| loss)
    }

    fn gradient(&self, data: &[Example]) -> Result<Gradient, &'static str> {
        self.loss_and_gradient(data).map(|(_, gradient)| gradient)
    }

    fn loss_and_gradient(&self, data: &[Example]) -> Result<(f64, Gradient), &'static str> {
        if data.is_empty() {
            return Err("training requires examples");
        }
        let n = data.len() as f64;
        let mut loss = 0.0;
        let mut gradient = Gradient {
            weights: [0.0; 2],
            bias: 0.0,
        };
        for &(features, target) in data {
            if features.iter().any(|value| !value.is_finite()) {
                return Err("features must be finite");
            }
            let logit = self.logit(features);
            loss += binary_cross_entropy_from_logit(logit, target)? / n;
            let error = sigmoid(logit) - target;
            gradient.weights[0] += error * features[0] / n;
            gradient.weights[1] += error * features[1] / n;
            gradient.bias += error / n;
        }
        if loss.is_finite()
            && gradient.bias.is_finite()
            && gradient.weights.iter().all(|value| value.is_finite())
        {
            Ok((loss, gradient))
        } else {
            Err("loss or gradient overflowed")
        }
    }

    #[cfg(test)]
    fn numerical_gradient(&self, data: &[Example]) -> Result<Gradient, &'static str> {
        const H: f64 = 1e-5;
        let mut gradient = Gradient {
            weights: [0.0; 2],
            bias: 0.0,
        };
        for (index, weight_gradient) in gradient.weights.iter_mut().enumerate() {
            let mut plus = *self;
            let mut minus = *self;
            plus.weights[index] += H;
            minus.weights[index] -= H;
            *weight_gradient = (plus.loss(data)? - minus.loss(data)?) / (2.0 * H);
        }
        let plus = Self {
            bias: self.bias + H,
            ..*self
        };
        let minus = Self {
            bias: self.bias - H,
            ..*self
        };
        gradient.bias = (plus.loss(data)? - minus.loss(data)?) / (2.0 * H);
        Ok(gradient)
    }

    fn step(self, data: &[Example], learning_rate: f64) -> Result<Self, &'static str> {
        validate_learning_rate(learning_rate)?;
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
        validate_learning_rate(learning_rate)?;
        for _ in 0..steps {
            self = self.step(data, learning_rate)?;
        }
        Ok(self)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = LogisticModel {
        weights: [0.0; 2],
        bias: 0.0,
    };
    let model = start.train(&TRAIN_DATA, 800, 0.2)?;
    println!("initial BCE: {:.6}", start.loss(&TRAIN_DATA)?);
    println!("trained BCE: {:.6}", model.loss(&TRAIN_DATA)?);
    for features in [[-1.0, -0.5], [0.1, 0.1], [1.2, 0.7]] {
        println!(
            "{features:?} -> logit {:.4}, probability {:.4}, class {}",
            model.logit(features),
            model.probability(features),
            model.predict(features)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_loss_handles_extreme_logits() -> Result<(), &'static str> {
        assert!(binary_cross_entropy_from_logit(1_000.0, 1.0)?.is_finite());
        assert!(binary_cross_entropy_from_logit(-1_000.0, 0.0)?.is_finite());
        assert!(
            (binary_cross_entropy_from_logit(0.0, 1.0)? - std::f64::consts::LN_2).abs() < 1e-12
        );
        assert!(binary_cross_entropy_from_logit(0.0, 2.0).is_err());
        Ok(())
    }

    #[test]
    fn score_probability_and_class_are_distinct() {
        let model = LogisticModel {
            weights: [1.2, 0.4],
            bias: -0.5,
        };
        let features = [2.0, -1.0];
        assert!((model.logit(features) - 1.5).abs() < 1e-12);
        assert!((model.probability(features) - 0.817_574_476_193_643_7).abs() < 1e-12);
        assert_eq!(model.predict(features), 1);
    }

    #[test]
    fn classifier_learns_the_fixture() -> Result<(), &'static str> {
        let start = LogisticModel {
            weights: [0.0; 2],
            bias: 0.0,
        };
        let model = start.train(&TRAIN_DATA, 800, 0.2)?;
        assert!(model.loss(&TRAIN_DATA)? < 0.04);
        assert!(TRAIN_DATA
            .iter()
            .all(|&(features, target)| model.predict(features) == target as u8));
        Ok(())
    }

    #[test]
    fn analytical_gradient_matches_stable_loss_difference() -> Result<(), &'static str> {
        let model = LogisticModel {
            weights: [0.3, -0.2],
            bias: 0.1,
        };
        let analytic = model.gradient(&TRAIN_DATA)?;
        let numerical = model.numerical_gradient(&TRAIN_DATA)?;
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 + 1e-4 * a.abs().max(b.abs());
        assert!(close(analytic.weights[0], numerical.weights[0]));
        assert!(close(analytic.weights[1], numerical.weights[1]));
        assert!(close(analytic.bias, numerical.bias));
        Ok(())
    }
}
