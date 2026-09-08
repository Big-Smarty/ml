fn rnn_state(_input: f64, _previous: f64, _wx: f64, _wh: f64, _bias: f64) -> f64 {
    // TODO: compute the next tanh state.
    todo!()
}
fn main() {
    println!("state={}", rnn_state(1.0, 0.0, 0.5, 0.2, 0.0));
}
#[test]
fn state_uses_current_and_previous_values() {
    assert!((rnn_state(2.0, -1.0, 0.5, 0.25, 0.1) - 0.6910694698).abs() < 1e-9);
}
