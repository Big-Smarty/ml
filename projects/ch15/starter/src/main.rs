fn gini_impurity(positive: usize, total: usize) -> f64 {
    assert!(total > 0 && positive <= total);
    let p = positive as f64 / total as f64;
    2.0 * p * (1.0 - p)
}
fn weighted_impurity(left: (usize, usize), right: (usize, usize)) -> f64 {
    // TODO: combine the children's Gini impurities, weighted by their row counts.
    let _ = (left, right);
    todo!("compute weighted child impurity")
}
fn main() {
    let _guided_step = weighted_impurity;
    let labels = [0, 0, 1];
    let positives = labels.iter().filter(|&&label| label == 1).count();
    println!(
        "parent Gini impurity: {:.3}",
        gini_impurity(positives, labels.len())
    );
}
#[test]
fn weighted_impurity_matches_examples() {
    assert_eq!(weighted_impurity((0, 3), (1, 1)), 0.0);
    assert!((weighted_impurity((1, 3), (0, 1)) - 1.0 / 3.0).abs() < 1e-12);
}
