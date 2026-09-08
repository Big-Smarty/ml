fn gini(positive: usize, total: usize) -> f64 {
    // TODO: implement node impurity.
    let _ = (positive, total);
    todo!("compute 2p(1-p)")
}
fn main() {
    let _guided_step = gini;
    let labels = [0, 0, 1];
    let positives = labels.iter().filter(|&&label| label == 1).count();
    println!(
        "majority class: {}",
        (positives * 2 >= labels.len()) as usize
    );
}
#[test]
fn pure_and_mixed() {
    assert_eq!(gini(0, 4), 0.0);
    assert!((gini(2, 4) - 0.5).abs() < 1e-12);
}
