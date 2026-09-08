fn dot(_user: &[f64], _item: &[f64]) -> f64 {
    // TODO: sum coordinate products.
    todo!()
}
fn main() {
    println!("score={}", dot(&[1.0, 2.0], &[3.0, 4.0]));
}
#[test]
fn score_uses_every_factor() {
    assert_eq!(dot(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
}
