fn q_update(
    old_value: f64,
    reward: f64,
    best_next_value: f64,
    done: bool,
    learning_rate: f64,
    discount_factor: f64,
) -> f64 {
    let target = reward
        + if done {
            0.0
        } else {
            discount_factor * best_next_value
        };
    old_value + learning_rate * (target - old_value)
}

fn main() {
    let _solved_update: fn(f64, f64, f64, bool, f64, f64) -> f64 = q_update;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_toward_bootstrapped_target() {
        assert!((q_update(0.2, 1.0, 0.5, false, 0.1, 0.9) - 0.325).abs() < 1e-12);
        assert!((q_update(0.2, 1.0, 99.0, true, 0.1, 0.9) - 0.28).abs() < 1e-12);
    }
}
