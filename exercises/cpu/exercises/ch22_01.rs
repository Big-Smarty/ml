fn reconstruction_mse(_input: &[f64], _output: &[f64]) -> f64 {
    // TODO: average squared coordinate errors.
    todo!()
}
fn main() {
    println!("mse={}", reconstruction_mse(&[1.0], &[0.0]));
}
#[test]
fn averages_over_coordinates() {
    assert_eq!(reconstruction_mse(&[1.0, 3.0], &[2.0, 1.0]), 2.5);
}
