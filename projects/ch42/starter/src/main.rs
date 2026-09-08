fn stable_softmax(scores: &[f32]) -> Vec<f32> {
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
    let denominator = probabilities.iter().sum::<f32>();
    probabilities
        .iter_mut()
        .for_each(|probability| *probability /= denominator);
    probabilities
}
fn online_softmax_state(_scores: &[f32]) -> (f32, f32) {
    // TODO: update the running maximum and shifted denominator for every score.
    todo!("implement online softmax state")
}
fn main() {
    let _guided: fn(&[f32]) -> (f32, f32) = online_softmax_state;
    println!(
        "checkpoint stable softmax: {:?}",
        stable_softmax(&[1.0, 3.0, 2.0])
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn online_state_matches_two_pass() {
        let x = [1.0, 3.0, 2.0];
        let (maximum, shifted_denominator) = online_softmax_state(&x);
        let probabilities = stable_softmax(&x);
        assert!((maximum - 3.0).abs() < 1e-6);
        assert!((shifted_denominator - 1.0 / probabilities[1]).abs() < 1e-6);
    }
}
