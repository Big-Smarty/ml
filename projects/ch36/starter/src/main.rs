fn residual_add(input: &[f32], residual_branch: &[f32]) -> Vec<f32> {
    // TODO: add the residual_branch to the matching input element.
    let _ = (input, residual_branch);
    todo!("guided repair: complete the residual connection")
}

fn causal_softmax(scores: &[f32], query_position: usize, positions: usize) -> Vec<f32> {
    let mut probabilities = vec![0.0; positions];
    let allowed =
        &scores[query_position * positions..query_position * positions + query_position + 1];
    let max = allowed.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = allowed.iter().map(|x| (x - max).exp()).sum::<f32>();
    for key_position in 0..=query_position {
        probabilities[key_position] =
            (scores[query_position * positions + key_position] - max).exp() / sum;
    }
    probabilities
}
fn weighted_value(probabilities: &[f32], values: &[f32]) -> f32 {
    probabilities
        .iter()
        .zip(values)
        .map(|(probability, value)| probability * value)
        .sum()
}

fn main() {
    let _exercise = residual_add as fn(&[f32], &[f32]) -> Vec<f32>;
    let probabilities = causal_softmax(&[0.0, 0.0, 1.0, 2.0], 1, 2);
    let attended = weighted_value(&probabilities, &[3.0, 7.0]);
    println!("Chapter 35 causal attention output: {attended:.3}");
    println!("Complete residual_add(), then run cargo test.");
}

#[test]
fn residual_connection_adds_identity_and_update() {
    assert_eq!(residual_add(&[1.0, -2.0], &[0.5, 3.0]), vec![1.5, 1.0]);
}
