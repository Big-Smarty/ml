fn regularized_weight_gradient(data_gradient: f64, weight: f64, l2: f64) -> f64 {
    data_gradient + 2.0 * l2 * weight
}
fn main() {
    println!("{}", regularized_weight_gradient(-3.0, 2.0, 0.1));
}
#[test]
fn regularization_pushes_toward_zero() {
    assert!((regularized_weight_gradient(-3.0, 2.0, 0.1) + 2.6).abs() < 1e-12);
    assert_eq!(regularized_weight_gradient(3.0, -2.0, 0.0), 3.0);
    assert!((regularized_weight_gradient(0.0, -2.0, 0.1) + 0.4).abs() < 1e-12);
}
