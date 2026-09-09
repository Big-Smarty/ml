// Return the analytical MSE derivative for one scalar weight.
fn mse_weight_gradient(weight: f64, data: &[(f64, f64)]) -> f64 {
    // TODO: compute the average analytical derivative.
    data.iter()
        .map(|(input, target)| 2.0 * (weight * input - target) * input)
        .sum::<f64>()
        / data.len() as f64
}
fn main() {
    println!("{}", mse_weight_gradient(0.0, &[(1.0, 2.0)]));
}

fn sim_an_gradient_delta(w: f64, b: f64, x: f64, y: f64, n: usize) -> (f64, f64) {
    let error = predict(w, b, x) - y;
    (2.0 * error * x / n as f64, 2.0 * error / n as f64)
}

fn predict(w: f64, b: f64, x: f64) -> f64 {
    w * x + b
}

#[test]
fn derivative_uses_every_example() {
    assert!((mse_weight_gradient(0.0, &[(-1.0, -2.0), (1.0, 2.0)]) + 4.0).abs() < 1e-12);
    assert_eq!(mse_weight_gradient(1.0, &[(1.0, 0.0), (2.0, 1.0)]), 3.0);
}
