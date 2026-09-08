fn reconstruction_mse(input: &[f64], output: &[f64]) -> f64 {
    input
        .iter()
        .zip(output)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / input.len() as f64
}
fn main() {
    println!("mse={}", reconstruction_mse(&[1.0], &[0.0]));
}
#[test]
fn averages_over_coordinates() {
    assert_eq!(reconstruction_mse(&[1.0, 3.0], &[2.0, 1.0]), 2.5);
}
