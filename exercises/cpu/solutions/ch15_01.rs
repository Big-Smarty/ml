fn gini(positive: usize, total: usize) -> f64 {
    assert!(total > 0 && positive <= total);
    let p = positive as f64 / total as f64;
    2.0 * p * (1.0 - p)
}
fn main() {
    println!("{}", gini(2, 4));
}
#[test]
fn impurity() {
    assert_eq!(gini(0, 4), 0.0);
    assert_eq!(gini(2, 4), 0.5);
}
