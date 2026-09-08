fn reconstruction_mse(reconstruction: &[f64], target: &[f64]) -> f64 {
    reconstruction
        .iter()
        .zip(target)
        .map(|(prediction, target)| (prediction - target).powi(2))
        .sum::<f64>()
        / target.len() as f64
}
fn main() {
    println!("mse={}", reconstruction_mse(&[1.0], &[0.0]));
}
#[test]
fn averages_over_coordinates() {
    assert_eq!(reconstruction_mse(&[1.0, 3.0], &[2.0, 1.0]), 2.5);
}
