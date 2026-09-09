fn dot(weights: &[f64], features: &[f64]) -> Option<f64> {
    // TODO: validate the lengths and compute the dot product.
    let _ = (weights, features);
    if weights.len() != features.len() {
        return None;
    }
    Some(
        weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum::<f64>(),
    )
}
fn main() {
    println!("{:?}", dot(&[2.0, -1.0], &[3.0, 4.0]));
}
#[test]
fn dot_checks_shape() {
    assert_eq!(dot(&[2.0, -1.0], &[3.0, 4.0]), Some(2.0));
    assert_eq!(dot(&[1.0], &[1.0, 2.0]), None);
}
