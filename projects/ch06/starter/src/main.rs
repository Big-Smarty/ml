fn regularized_weight_gradient(data_gradient: f64, weight: f64, lambda: f64) -> f64 {
    let _ = (data_gradient, weight, lambda);
    todo!("add the derivative of lambda * weight^2")
}

fn main() {
    let _guided_todo: fn(f64, f64, f64) -> f64 = regularized_weight_gradient;
    println!("prior checkpoint: unregularized data gradient is -3");
    println!("Run cargo test to implement the new L2-gradient TODO.");
}

#[test]
fn l2_adds_two_lambda_weight() {
    assert!((regularized_weight_gradient(-3.0, 2.0, 0.1) + 2.6).abs() < 1e-12);
    assert_eq!(regularized_weight_gradient(3.0, -2.0, 0.0), 3.0);
    assert!((regularized_weight_gradient(0.0, -2.0, 0.1) + 0.4).abs() < 1e-12);
}
