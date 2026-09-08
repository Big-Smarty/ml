fn precision(true_positive: usize, false_positive: usize) -> f64 {
    let _ = (true_positive, false_positive);
    todo!("return TP / (TP + FP), or 0 when there are no predicted positives")
}

fn main() {
    let _guided_todo: fn(usize, usize) -> f64 = precision;
    println!("prior checkpoint: score 0.7 at threshold 0.5 predicts positive");
    println!("Run cargo test to implement the new precision TODO.");
}

#[test]
fn precision_handles_counts_and_empty_denominator() {
    assert!((precision(3, 1) - 0.75).abs() < 1e-12);
    assert_eq!(precision(0, 0), 0.0);
}
