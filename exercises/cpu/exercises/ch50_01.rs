// Each tuple is (first weight-gradient sum, example count) from one worker.
fn mean_first_weight_gradient(parts: &[(f64, usize)]) -> f64 {
    // TODO: divide the combined gradient sum by the combined example count.
    let _ = parts;
    todo!("reduce the first weight coordinate by examples")
}
fn main() {
    println!("Aggregate sufficient statistics, not worker means.");
}
#[test]
fn uneven_workers() {
    assert_eq!(mean_first_weight_gradient(&[(4.0, 1), (6.0, 3)]), 2.5);
}
