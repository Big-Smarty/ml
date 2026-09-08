fn l2_gradient(data_gradient: f64, weight: f64, lambda: f64) -> f64 {
    // TODO: add the regularization derivative.
    let _ = (data_gradient, weight, lambda);
    todo!("differentiate lambda * weight squared and add it")
}
fn main() {
    println!("{}", l2_gradient(-3.0, 2.0, 0.1));
}
#[test]
fn regularization_pushes_toward_zero() {
    assert!((l2_gradient(-3.0, 2.0, 0.1) + 2.6).abs() < 1e-12);
    assert_eq!(l2_gradient(3.0, -2.0, 0.0), 3.0);
    assert!((l2_gradient(0.0, -2.0, 0.1) + 0.4).abs() < 1e-12);
}
