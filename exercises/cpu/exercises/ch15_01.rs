fn gini_impurity(positive: usize, total: usize) -> f64 {
    // TODO: compute class impurity 2p(1-p).
    let _ = (positive, total);
    todo!("compute class impurity 2p(1-p)")
}
fn main() {
    println!("{}", gini_impurity(2, 4));
}
#[test]
fn impurity() {
    assert_eq!(gini_impurity(0, 4), 0.0);
    assert_eq!(gini_impurity(2, 4), 0.5);
}
