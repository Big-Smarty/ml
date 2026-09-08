fn brier(probabilities: &[f64], labels: &[u8]) -> f64 {
    assert!(!probabilities.is_empty() && probabilities.len() == labels.len());
    probabilities
        .iter()
        .zip(labels)
        .map(|(p, y)| (p - *y as f64).powi(2))
        .sum::<f64>()
        / probabilities.len() as f64
}
fn main() {
    println!("{}", brier(&[0.2, 0.8], &[0, 1]));
}
#[test]
fn score() {
    assert!((brier(&[0.2, 0.8], &[0, 1]) - 0.04).abs() < 1e-12);
    assert!((brier(&[0.2], &[1]) - 0.64).abs() < 1e-12);
}
