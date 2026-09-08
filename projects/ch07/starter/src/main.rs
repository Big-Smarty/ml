fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_slope(output: f64) -> f64 {
    let _ = output;
    todo!("return the derivative using the already-computed sigmoid output")
}

fn main() {
    let _guided: fn(f64) -> f64 = sigmoid_slope;
    println!("prior checkpoint: sigmoid(0) = {}", sigmoid(0.0));
}

#[test]
fn slope_at_origin() {
    assert!((sigmoid_slope(sigmoid(0.0)) - 0.25).abs() < 1e-12);
    assert_eq!(sigmoid_slope(0.0), 0.0);
    assert!((sigmoid_slope(0.8) - 0.16).abs() < 1e-12);
}
