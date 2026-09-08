fn squared_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    // TODO: implement the complete two-coordinate distance.
    let _ = (a, b);
    todo!("sum the squared coordinate differences")
}
fn main() {
    let _guided_step = squared_distance;
    let (a, b) = ([1.0, 2.0], [4.0, 6.0]);
    println!(
        "first-coordinate squared difference: {}",
        (a[0] - b[0]) * (a[0] - b[0])
    );
}
#[test]
fn three_four_five_triangle() {
    assert_eq!(squared_distance([1.0, 2.0], [4.0, 6.0]), 25.0);
    assert_eq!(squared_distance([2.0, -1.0], [2.0, -1.0]), 0.0);
    assert_eq!(squared_distance([2.0, -1.0], [-1.0, -1.0]), 9.0);
}
