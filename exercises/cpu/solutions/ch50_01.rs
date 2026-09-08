// Each tuple is (first weight-gradient sum, example count) from one worker.
fn mean_first_weight_gradient(parts: &[(f64, usize)]) -> f64 {
    let sum: f64 = parts.iter().map(|(sum, _)| sum).sum();
    let count: usize = parts.iter().map(|(_, count)| count).sum();
    sum / count as f64
}
fn main() {
    println!("Aggregate sufficient statistics, not worker means.");
}
#[test]
fn uneven_workers() {
    assert_eq!(mean_first_weight_gradient(&[(4.0, 1), (6.0, 3)]), 2.5);
}
