fn residual_add(input: &[f64], residual_branch: &[f64]) -> Vec<f64> {
    input
        .iter()
        .zip(residual_branch)
        .map(|(input, branch)| input + branch)
        .collect()
}
fn main() {
    println!("{:?}", residual_add(&[1.0], &[0.5]));
}
#[test]
fn residual_add_is_elementwise() {
    assert_eq!(residual_add(&[1.0, 2.0], &[0.5, -1.0]), [1.5, 1.0]);
}
