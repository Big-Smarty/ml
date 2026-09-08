// Return the analytical MSE derivative for one scalar weight.
fn mse_weight_gradient(weight: f64, data: &[(f64, f64)]) -> f64 {
    // TODO: compute the average analytical derivative.
    let _ = (weight, data);
    todo!("average 2 * (weight * x - y) * x")
}
fn main() {
    println!("{}", mse_weight_gradient(0.0, &[(1.0, 2.0)]));
}
#[test]
fn derivative_uses_every_example() {
    assert!((mse_weight_gradient(0.0, &[(-1.0, -2.0), (1.0, 2.0)]) + 4.0).abs() < 1e-12);
    assert_eq!(mse_weight_gradient(1.0, &[(1.0, 0.0), (2.0, 1.0)]), 3.0);
}
