fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}
fn main() {
    println!("Every fairness rate needs its denominator named.");
}
#[test]
fn defined_and_undefined_rates() {
    assert_eq!(rate(3, 4), Some(0.75));
    assert_eq!(rate(0, 0), None);
}
