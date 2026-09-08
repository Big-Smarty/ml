fn causal_softmax(scores: &[f64], query_position: usize, positions: usize) -> Vec<f64> {
    let mut probabilities = vec![0.0; positions];
    let allowed =
        &scores[query_position * positions..query_position * positions + query_position + 1];
    let max = allowed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum = allowed.iter().map(|x| (x - max).exp()).sum::<f64>();
    for key_position in 0..=query_position {
        probabilities[key_position] =
            (scores[query_position * positions + key_position] - max).exp() / sum;
    }
    probabilities
}
fn weighted_value(probabilities: &[f64], values: &[f64]) -> f64 {
    // TODO: return the dot product.
    let _ = (probabilities, values);
    todo!("guided repair: combine attention weights and values")
}
fn main() {
    let _exercise = weighted_value as fn(&[f64], &[f64]) -> f64;
    let probabilities = causal_softmax(&[1.0, 99.0, 0.0, 0.0], 0, 2);
    println!("first row causal probabilities: {probabilities:?}");
    println!("Complete weighted_value, then run cargo test.");
}
#[test]
fn combines_values() {
    assert!((weighted_value(&[0.25, 0.75], &[2.0, 6.0]) - 5.0).abs() < 1e-12);
}
