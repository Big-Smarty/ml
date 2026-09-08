fn squared_distance(features: [f64; 2], other_features: [f64; 2]) -> f64 {
    features
        .into_iter()
        .zip(other_features)
        .map(|(value, other_value)| (value - other_value).powi(2))
        .sum()
}
fn main() {
    println!("{}", squared_distance([1., 2.], [4., 6.]));
}
#[test]
fn distance() {
    assert_eq!(squared_distance([1., 2.], [4., 6.]), 25.0);
    assert_eq!(squared_distance([2., -1.], [2., -1.]), 0.0);
    assert_eq!(squared_distance([2., -1.], [-1., -1.]), 9.0);
}
