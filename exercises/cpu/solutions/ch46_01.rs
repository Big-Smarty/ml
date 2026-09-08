fn dpo_loss(policy_margin: f64, reference_margin: f64, beta: f64) -> f64 {
    (1.0 + (-beta * (policy_margin - reference_margin)).exp()).ln()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_margins_give_log_two() {
        assert!((dpo_loss(0.4, 0.4, 0.5) - std::f64::consts::LN_2).abs() < 1e-12);
        assert!((dpo_loss(0.8, 0.4, 0.5) - 0.5981388693815918).abs() < 1e-12);
        assert!(dpo_loss(-1.0, 0.4, 0.5) > std::f64::consts::LN_2);
    }
}
