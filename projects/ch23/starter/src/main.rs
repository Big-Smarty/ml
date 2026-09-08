fn cosine(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb)
}
fn rnn_state(
    _input: f64,
    _previous_state: f64,
    _input_weight: f64,
    _recurrent_weight: f64,
    _bias: f64,
) -> f64 {
    // TODO: combine the current input and previous state, then apply tanh.
    todo!("tanh(input_weight*input + recurrent_weight*previous_state + bias)")
}
fn main() {
    let _guided_step = rnn_state;
    println!(
        "previous checkpoint cosine: {:.3}",
        cosine(&[1.0, 0.0], &[0.8, 0.2])
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recurrence_uses_previous_state() {
        let expected = (0.5_f64 * 2.0 - 0.25 + 0.1).tanh();
        assert!((rnn_state(2.0, -1.0, 0.5, 0.25, 0.1) - expected).abs() < 1e-12);
    }
}
