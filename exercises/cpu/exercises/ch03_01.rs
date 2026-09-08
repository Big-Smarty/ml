fn dot(weights: &[f64], features: &[f64]) -> Option<f64> {
    // TODO: validate the lengths and compute the dot product.
    let _ = (weights, features);
    todo!("return None for unequal lengths, otherwise sum matching products")
}
fn main() {
    println!("{:?}", dot(&[2.0, -1.0], &[3.0, 4.0]));
}
#[test]
fn dot_checks_shape() {
    assert_eq!(dot(&[2.0, -1.0], &[3.0, 4.0]), Some(2.0));
    assert_eq!(dot(&[1.0], &[1.0, 2.0]), None);
}
