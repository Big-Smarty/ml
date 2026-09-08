fn update_online_softmax_state(
    running_maximum: f32,
    shifted_denominator: f32,
    score: f32,
) -> (f32, f32) {
    let next_maximum = running_maximum.max(score);
    (
        next_maximum,
        shifted_denominator * (running_maximum - next_maximum).exp() + (score - next_maximum).exp(),
    )
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
