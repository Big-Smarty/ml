//! Binary logistic regression trained directly from stable logits.

const DATA: [([f64; 2], f64); 8] = [
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

impl LogisticModel {
    fn logit(self, x: [f64; 2]) -> f64 {
        self.weights[0] * x[0] + self.weights[1] * x[1] + self.bias
    }

    fn probability(self, x: [f64; 2]) -> f64 {
        sigmoid(self.logit(x))
    }

    fn loss_and_gradient(self, data: &[([f64; 2], f64)]) -> Result<(f64, Self), &'static str> {
        if data.is_empty() {
            return Err("training requires examples");
        }
        let n = data.len() as f64;
        let mut loss = 0.0;
        let mut gradient = Self {
            weights: [0.0; 2],
            bias: 0.0,
        };
        for &(x, y) in data {
            if x.iter().any(|value| !value.is_finite()) {
                return Err("features must be finite");
            }
            let logit = self.logit(x);
            loss += binary_cross_entropy_from_logit(logit, y)? / n;
            let error = sigmoid(logit) - y;
            gradient.weights[0] += error * x[0] / n;
            gradient.weights[1] += error * x[1] / n;
            gradient.bias += error / n;
        }
        Ok((loss, gradient))
    }

    fn train(
        mut self,
        data: &[([f64; 2], f64)],
        steps: usize,
        rate: f64,
    ) -> Result<Self, &'static str> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        for _ in 0..steps {
            let (_, gradient) = self.loss_and_gradient(data)?;
            self.weights[0] -= rate * gradient.weights[0];
            self.weights[1] -= rate * gradient.weights[1];
            self.bias -= rate * gradient.bias;
        }
        Ok(self)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = LogisticModel {
        weights: [0.0; 2],
        bias: 0.0,
    };
    let model = start.train(&DATA, 800, 0.2)?;
    println!("initial BCE: {:.6}", start.loss_and_gradient(&DATA)?.0);
    println!("trained BCE: {:.6}", model.loss_and_gradient(&DATA)?.0);
    for point in [[-1.0, -0.5], [0.1, 0.1], [1.2, 0.7]] {
        println!(
            "{point:?} -> probability {:.4}, class {}",
            model.probability(point),
            u8::from(model.probability(point) >= 0.5)
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
    fn classifier_learns_the_fixture() -> Result<(), &'static str> {
        let start = LogisticModel {
            weights: [0.0; 2],
            bias: 0.0,
        };
        let model = start.train(&DATA, 800, 0.2)?;
        assert!(model.loss_and_gradient(&DATA)?.0 < 0.04);
        assert!(DATA
            .iter()
            .all(|&(x, y)| (model.probability(x) >= 0.5) == (y == 1.0)));
        Ok(())
    }

    #[test]
    fn analytical_gradient_matches_stable_loss_difference() -> Result<(), &'static str> {
        const H: f64 = 1e-5;
        let model = LogisticModel {
            weights: [0.3, -0.2],
            bias: 0.1,
        };
        let analytic = model.loss_and_gradient(&DATA)?.1;
        let loss = |candidate: LogisticModel| candidate.loss_and_gradient(&DATA).map(|pair| pair.0);
        let dw0 = (loss(LogisticModel {
            weights: [model.weights[0] + H, model.weights[1]],
            ..model
        })? - loss(LogisticModel {
            weights: [model.weights[0] - H, model.weights[1]],
            ..model
        })?) / (2.0 * H);
        let dw1 = (loss(LogisticModel {
            weights: [model.weights[0], model.weights[1] + H],
            ..model
        })? - loss(LogisticModel {
            weights: [model.weights[0], model.weights[1] - H],
            ..model
        })?) / (2.0 * H);
        let db = (loss(LogisticModel {
            bias: model.bias + H,
            ..model
        })? - loss(LogisticModel {
            bias: model.bias - H,
            ..model
        })?) / (2.0 * H);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 + 1e-4 * a.abs().max(b.abs());
        assert!(close(analytic.weights[0], dw0));
        assert!(close(analytic.weights[1], dw1));
        assert!(close(analytic.bias, db));
        Ok(())
    }
}
