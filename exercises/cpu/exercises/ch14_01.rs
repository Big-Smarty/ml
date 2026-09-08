fn squared_distance(features: [f64; 2], other_features: [f64; 2]) -> f64 {
    // TODO: sum the squared coordinate differences.
    let _ = (features, other_features);
    todo!("zip coordinates and sum squared differences")
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
