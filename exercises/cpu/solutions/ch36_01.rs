fn residual(x: &[f32], update: &[f32]) -> Vec<f32> {
    x.iter().zip(update).map(|(a, b)| a + b).collect()
}
fn main() {
    println!("Residual paths carry identity and learned updates.");
}
#[test]
fn adds_paths() {
    assert_eq!(residual(&[1.0, 2.0], &[-0.5, 3.0]), vec![0.5, 5.0]);
}
