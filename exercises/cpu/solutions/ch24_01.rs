fn dot(user: &[f64], item: &[f64]) -> f64 {
    user.iter().zip(item).map(|(u, i)| u * i).sum()
}
fn main() {
    println!("score={}", dot(&[1.0, 2.0], &[3.0, 4.0]));
}
#[test]
fn score_uses_every_factor() {
    assert_eq!(dot(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
}
