fn mse_weight_gradient(weight: f64, data: &[(f64, f64)]) -> f64 {
    data.iter()
        .map(|&(x, y)| 2.0 * (weight * x - y) * x)
        .sum::<f64>()
        / data.len() as f64
}
fn main() {
    println!("{}", mse_weight_gradient(0.0, &[(1.0, 2.0)]));
}
#[test]
fn derivative_uses_every_example() {
    assert!((mse_weight_gradient(0.0, &[(-1.0, -2.0), (1.0, 2.0)]) + 4.0).abs() < 1e-12);
    assert_eq!(mse_weight_gradient(1.0, &[(1.0, 0.0), (2.0, 1.0)]), 3.0);
}
