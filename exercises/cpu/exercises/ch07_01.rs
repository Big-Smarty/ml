// TODO: return the sigmoid derivative using its output.
fn sigmoid_derivative_from_output(output: f64) -> f64 {
    let _ = output;
    todo!("use the sigmoid output")
}
fn main() {
    println!("{}", sigmoid_derivative_from_output(0.5));
}
#[test]
fn derivative() {
    assert_eq!(sigmoid_derivative_from_output(0.5), 0.25);
    assert_eq!(sigmoid_derivative_from_output(0.0), 0.0);
    assert!((sigmoid_derivative_from_output(0.8) - 0.16).abs() < 1e-12);
}
