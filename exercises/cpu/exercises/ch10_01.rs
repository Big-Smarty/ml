// TODO: compute stable log-sum-exp.
fn log_sum_exp(xs: &[f64]) -> f64 {
    let _ = xs;
    todo!("subtract the maximum before exponentiating")
}
fn main() {
    println!("{}", log_sum_exp(&[1.0, 2.0]));
}
#[test]
fn stable() {
    let x = log_sum_exp(&[1000.0, 1001.0]);
    assert!(x.is_finite() && (x - 1001.3132616875).abs() < 1e-9);
    assert!((log_sum_exp(&[0.0, 0.0]) - std::f64::consts::LN_2).abs() < 1e-12);
}
