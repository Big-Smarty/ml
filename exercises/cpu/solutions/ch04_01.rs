fn sigmoid(logit: f64) -> f64 {
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let e = logit.exp();
        e / (1.0 + e)
    }
}
fn main() {
    println!("{}", sigmoid(0.0));
}
#[test]
fn sigmoid_is_stable() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
    assert!(sigmoid(-1000.0).is_finite());
    assert!(sigmoid(1000.0).is_finite());
    assert!((sigmoid(2.0) - 0.8807970779778823).abs() < 1e-12);
    assert!((sigmoid(-2.0) - 0.11920292202211755).abs() < 1e-12);
}
