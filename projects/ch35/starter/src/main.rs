fn causal_softmax(scores: &[f64], row: usize, t: usize) -> Vec<f64> {
    let mut out = vec![0.0; t];
    let allowed = &scores[row * t..row * t + row + 1];
    let max = allowed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum = allowed.iter().map(|x| (x - max).exp()).sum::<f64>();
    for j in 0..=row {
        out[j] = (scores[row * t + j] - max).exp() / sum;
    }
    out
}
fn weighted_value(weights: &[f64], values: &[f64]) -> f64 {
    // TODO: return the dot product.
    let _ = (weights, values);
    todo!("guided repair: combine attention weights and values")
}
fn main() {
    let _exercise = weighted_value as fn(&[f64], &[f64]) -> f64;
    let w = causal_softmax(&[1.0, 99.0, 0.0, 0.0], 0, 2);
    println!("first row causal weights: {w:?}");
    println!("Complete weighted_value, then run cargo test.");
}
#[test]
fn combines_values() {
    assert!((weighted_value(&[0.25, 0.75], &[2.0, 6.0]) - 5.0).abs() < 1e-12);
}
