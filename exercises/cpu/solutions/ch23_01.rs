fn rnn_state(
    input: f64,
    previous_state: f64,
    input_weight: f64,
    recurrent_weight: f64,
    bias: f64,
) -> f64 {
    (input_weight * input + recurrent_weight * previous_state + bias).tanh()
}
fn main() {
    println!("state={}", rnn_state(1.0, 0.0, 0.5, 0.2, 0.0));
}
#[test]
fn state_uses_current_and_previous_values() {
    assert!((rnn_state(2.0, -1.0, 0.5, 0.25, 0.1) - 0.6910694698).abs() < 1e-9);
}
