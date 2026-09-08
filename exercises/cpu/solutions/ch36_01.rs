fn residual_add(input: &[f32], residual_branch: &[f32]) -> Vec<f32> {
    input
        .iter()
        .zip(residual_branch)
        .map(|(a, b)| a + b)
        .collect()
}
fn main() {
    println!("A residual connection adds the identity shortcut and learned branch.");
}
#[test]
fn residual_connection_adds_values() {
    assert_eq!(residual_add(&[1.0, 2.0], &[-0.5, 3.0]), vec![0.5, 5.0]);
}
