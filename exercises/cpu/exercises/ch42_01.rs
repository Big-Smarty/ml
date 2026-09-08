fn update_online_softmax_state(
    _running_maximum: f32,
    _shifted_denominator: f32,
    _score: f32,
) -> (f32, f32) {
    // TODO: rescale and update the online-softmax state with this score.
    todo!("online softmax state update")
}
fn main() {
    println!("state: {:?}", update_online_softmax_state(1.0, 1.0, 3.0));
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rescales_old_terms_when_max_changes() {
        let (maximum, shifted_denominator) = update_online_softmax_state(1.0, 1.0, 3.0);
        assert!((maximum - 3.0).abs() < 1e-6);
        assert!((shifted_denominator - (1.0 + (-2.0f32).exp())).abs() < 1e-6);
    }
}
