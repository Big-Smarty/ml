fn predict(weight: f64, bias: f64, input: f64) -> f64 {
    weight * input + bias
}
fn main() {
    println!("{}", predict(2.0, 1.0, 3.0));
}
#[test]
fn prediction_uses_all_three_values() {
    assert_eq!(predict(2.0, 1.0, 3.0), 7.0);
    assert_eq!(predict(-1.0, 4.0, 2.0), 2.0);
}
