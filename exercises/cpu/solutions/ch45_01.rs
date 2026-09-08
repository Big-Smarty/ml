fn q_update(old: f64, reward: f64, best_next: f64, alpha: f64, gamma: f64) -> f64 {
    old + alpha * (reward + gamma * best_next - old)
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_toward_bootstrapped_target() {
        assert!((q_update(0.2, 1.0, 0.5, 0.1, 0.9) - 0.325).abs() < 1e-12);
    }
}
