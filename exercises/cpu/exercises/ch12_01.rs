fn brier(probabilities: &[f64], labels: &[u8]) -> f64 {
    // TODO: return the mean squared probability error.
    let _ = (probabilities, labels);
    todo!("return the mean squared probability error")
}
fn main() {
    println!("{}", brier(&[0.2, 0.8], &[0, 1]));
}
#[test]
fn score() {
    assert!((brier(&[0.2, 0.8], &[0, 1]) - 0.04).abs() < 1e-12);
    assert!((brier(&[0.2], &[1]) - 0.64).abs() < 1e-12);
}
