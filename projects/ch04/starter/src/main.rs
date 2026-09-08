fn stable_bce(logit: f64, target: f64) -> f64 {
    let _ = (logit, target);
    todo!("use max(logit, 0) - logit * target + ln(1 + exp(-abs(logit)))")
}

fn main() {
    let _guided_todo: fn(f64, f64) -> f64 = stable_bce;
    let weights = [1.2, 0.4];
    let features = [2.0, -1.0];
    let logit = weights[0] * features[0] + weights[1] * features[1] - 0.5;
    println!("prior checkpoint: logit={logit}");
    println!("Run cargo test to implement the new stable-BCE TODO.");
}

#[test]
fn loss_is_finite_for_confident_predictions() {
    assert!(stable_bce(1_000.0, 1.0).is_finite());
    assert!(stable_bce(-1_000.0, 0.0).is_finite());
    assert!((stable_bce(0.0, 1.0) - std::f64::consts::LN_2).abs() < 1e-12);
    assert!((stable_bce(1000.0, 0.0) - 1000.0).abs() < 1e-12);
    assert!((stable_bce(-1000.0, 1.0) - 1000.0).abs() < 1e-12);
}
