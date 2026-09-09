use std::f64;

fn sigmoid(logit: f64) -> f64 {
    // TODO: implement both stable branches.
    let _ = logit;
    if logit.is_sign_positive() {
        1.0 / (1.0 + f64::consts::E.powf(-logit))
    } else {
        let exp = logit.exp();
        exp / (1.0 + exp)
    }
}
fn main() {
    println!("{}", sigmoid(0.0));
    println!("{}", sigmoid(2.0));
    println!("{}", sigmoid(-2.0));
}
#[test]
fn sigmoid_is_stable() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
    assert!(sigmoid(-1000.0).is_finite());
    assert!(sigmoid(1000.0).is_finite());
    assert!((sigmoid(2.0) - 0.8807970779778823).abs() < 1e-12);
    assert!((sigmoid(-2.0) - 0.11920292202211755).abs() < 1e-12);
}
