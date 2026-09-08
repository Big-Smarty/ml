fn dot(weights: &[f64], features: &[f64]) -> Option<f64> {
    (weights.len() == features.len())
        .then(|| weights.iter().zip(features).map(|(w, x)| w * x).sum())
}
fn main() {
    println!("{:?}", dot(&[2.0, -1.0], &[3.0, 4.0]));
}
#[test]
fn dot_checks_shape() {
    assert_eq!(dot(&[2.0, -1.0], &[3.0, 4.0]), Some(2.0));
    assert_eq!(dot(&[1.0], &[1.0, 2.0]), None);
}
