fn residual(input: &[f32], update: &[f32]) -> Vec<f32> {
    // TODO: add the update to the matching input element.
    let _ = (input, update);
    todo!("guided repair: add the residual paths")
}

fn causal_scalar_attention(scores: &[f32], values: &[f32]) -> f32 {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = scores.iter().map(|x| (*x - max).exp()).collect();
    let sum = weights.iter().sum::<f32>();
    weights.iter().zip(values).map(|(w, v)| w / sum * v).sum()
}

fn main() {
    let _exercise = residual as fn(&[f32], &[f32]) -> Vec<f32>;
    let attended = causal_scalar_attention(&[1.0, 2.0], &[3.0, 7.0]);
    println!("Chapter 35 causal attention output: {attended:.3}");
    println!("Complete residual(), then run cargo test.");
}

#[test]
fn residual_path_preserves_and_updates() {
    assert_eq!(residual(&[1.0, -2.0], &[0.5, 3.0]), vec![1.5, 1.0]);
}
