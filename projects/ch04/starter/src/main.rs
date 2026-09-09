#[derive(Clone, Copy)]
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
    // TODO: validate the inputs, then implement stable BCE from the logit.
    let _ = (logit, target);
    if 0.0 > target || 1.0 < target {
        return Err("");
    }
    Ok(logit.max(0.0) - logit * target + (1.0 + (-logit.abs()).exp()).ln())
}

impl LogisticModel {
    fn logit(&self, features: [f64; 2]) -> f64 {
        self.weights[0] * features[0] + self.weights[1] * features[1] + self.bias
    }

    fn probability(&self, features: [f64; 2]) -> f64 {
        sigmoid(self.logit(features))
    }

    fn predict(&self, features: [f64; 2]) -> u8 {
        u8::from(self.probability(features) >= 0.5)
    }
}

fn main() {
    let _guided_todo: fn(f64, f64) -> Result<f64, &'static str> = binary_cross_entropy_from_logit;
    let model = LogisticModel {
        weights: [1.2, 0.4],
        bias: -0.5,
    };
    let features = [2.0, -1.0];
    println!(
        "prior checkpoint: logit={:.1}, probability={:.4}, class={}",
        model.logit(features),
        model.probability(features),
        model.predict(features)
    );
    println!("Run cargo test to implement the new stable-BCE TODO.");
}

#[test]
fn loss_is_stable_and_checks_targets() -> Result<(), &'static str> {
    assert!(binary_cross_entropy_from_logit(1_000.0, 1.0)?.is_finite());
    assert!(binary_cross_entropy_from_logit(-1_000.0, 0.0)?.is_finite());
    assert!((binary_cross_entropy_from_logit(0.0, 1.0)? - std::f64::consts::LN_2).abs() < 1e-12);
    assert!((binary_cross_entropy_from_logit(1_000.0, 0.0)? - 1_000.0).abs() < 1e-12);
    assert!(binary_cross_entropy_from_logit(0.0, 2.0).is_err());
    Ok(())
}
