fn dot(user_embedding: &[f64], item_embedding: &[f64]) -> f64 {
    user_embedding
        .iter()
        .zip(item_embedding)
        .map(|(user_factor, item_factor)| user_factor * item_factor)
        .sum()
}
fn main() {
    println!("score={}", dot(&[1.0, 2.0], &[3.0, 4.0]));
}
#[test]
fn dot_uses_every_factor() {
    assert_eq!(dot(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
}
