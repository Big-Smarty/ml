fn precision(true_positive: usize, false_positive: usize) -> f64 {
    let _ = (true_positive, false_positive);
    todo!("return TP / (TP + FP), or 0 when there are no predicted positives")
}

fn predict(probability: f64, threshold: f64) -> bool {
    probability >= threshold
}

fn main() {
    let _guided_todo: fn(usize, usize) -> f64 = precision;
    let probability = 0.7;
    let threshold = 0.5;
    println!(
        "prior checkpoint: probability {probability:.1} at threshold {threshold:.1} predicts class {}",
        u8::from(predict(probability, threshold))
    );
    println!("Run cargo test to implement the new precision TODO.");
}

#[test]
fn precision_handles_counts_and_empty_denominator() {
    assert!((precision(3, 1) - 0.75).abs() < 1e-12);
    assert_eq!(precision(0, 0), 0.0);
}
