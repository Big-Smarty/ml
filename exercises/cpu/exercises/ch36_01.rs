fn residual(x: &[f32], update: &[f32]) -> Vec<f32> {
    // TODO: add matching elements.
    let _ = (x, update);
    todo!()
}
fn main() {
    println!("Residual paths carry identity and learned updates.");
}
#[test]
fn adds_paths() {
    assert_eq!(residual(&[1.0, 2.0], &[-0.5, 3.0]), vec![0.5, 5.0]);
}
