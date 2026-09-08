fn squared_distance(features: [f64; 2], other_features: [f64; 2]) -> f64 {
    // TODO: implement the complete two-coordinate distance.
    let _ = (features, other_features);
    todo!("sum the squared coordinate differences")
}
fn main() {
    let _guided_step = squared_distance;
    let (features, other_features) = ([1.0, 2.0], [4.0, 6.0]);
    println!(
        "first-coordinate squared difference: {}",
        (features[0] - other_features[0]) * (features[0] - other_features[0])
    );
}
#[test]
fn three_four_five_triangle() {
    assert_eq!(squared_distance([1.0, 2.0], [4.0, 6.0]), 25.0);
    assert_eq!(squared_distance([2.0, -1.0], [2.0, -1.0]), 0.0);
    assert_eq!(squared_distance([2.0, -1.0], [-1.0, -1.0]), 9.0);
}
