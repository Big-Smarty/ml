fn residual(_input: &[f64], _branch: &[f64]) -> Vec<f64> {
    // TODO: add matching elements.
    todo!()
}
fn main() {
    println!("{:?}", residual(&[1.0], &[0.5]));
}
#[test]
fn identity_shortcut_is_elementwise() {
    assert_eq!(residual(&[1.0, 2.0], &[0.5, -1.0]), [1.5, 1.0]);
}
