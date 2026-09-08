fn recall(true_positive: usize, false_negative: usize) -> f64 {
    let total = true_positive + false_negative;
    if total == 0 {
        0.0
    } else {
        true_positive as f64 / total as f64
    }
}
fn main() {
    println!("{}", recall(3, 1));
}
#[test]
fn recall_counts_missed_positives() {
    assert!((recall(3, 1) - 0.75).abs() < 1e-12);
    assert_eq!(recall(0, 0), 0.0);
}
