const DATA: [(f64, f64); 3] = [(-1.0, -1.0), (0.0, 1.0), (1.0, 3.0)];

fn gradient(weight: f64, bias: f64) -> (f64, f64) {
    let _ = (weight, bias);
    todo!("average 2 * error * x and 2 * error over DATA")
}

fn main() {
    let _guided_todo: fn(f64, f64) -> (f64, f64) = gradient;
    let _ = DATA.len();
    println!("prior checkpoint: prediction at x=2 is {}", 0.5 * 2.0 + 1.0);
    println!("Run cargo test to implement the new analytical-gradient TODO.");
}

#[test]
fn gradient_at_zero() {
    let (dw, db) = gradient(0.0, 0.0);
    assert!((dw + 8.0 / 3.0).abs() < 1e-12);
    assert!((db + 2.0).abs() < 1e-12);
    let (dw, db) = gradient(1.0, 0.5);
    assert!((dw + 4.0 / 3.0).abs() < 1e-12);
    assert!((db + 1.0).abs() < 1e-12);
}
