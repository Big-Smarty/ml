fn recall(true_positive: usize, false_negative: usize) -> f64 {
    // TODO: calculate recall safely.
    let _ = (true_positive, false_negative);
    todo!("return TP / (TP + FN), or zero when the denominator is zero")
}
fn main() {
    println!("{}", recall(3, 1));
}
#[test]
fn recall_counts_missed_positives() {
    assert!((recall(3, 1) - 0.75).abs() < 1e-12);
    assert_eq!(recall(0, 0), 0.0);
}
