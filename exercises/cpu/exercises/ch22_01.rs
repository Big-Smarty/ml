fn reconstruction_mse(_reconstruction: &[f64], _target: &[f64]) -> f64 {
    // TODO: average squared coordinate errors, with reconstruction before target.
    todo!()
}
fn main() {
    println!("mse={}", reconstruction_mse(&[1.0], &[0.0]));
}
#[test]
fn averages_over_coordinates() {
    assert_eq!(reconstruction_mse(&[1.0, 3.0], &[2.0, 1.0]), 2.5);
}
