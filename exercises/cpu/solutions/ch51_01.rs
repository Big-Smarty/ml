fn sample(mean: f64, log_variance: f64, epsilon: f64) -> f64 {
    mean + (0.5 * log_variance).exp() * epsilon
}
fn main() {
    println!("Separate randomness from differentiable parameters.");
}
#[test]
fn unit_variance() {
    assert!((sample(1.0, 0.0, 0.25) - 1.25).abs() < 1e-12);
}

#[test]
fn quarter_variance_halves_noise_scale() {
    assert!((sample(0.8, 0.25_f64.ln(), -0.5) - 0.55).abs() < 1e-12);
}
