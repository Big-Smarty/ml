fn dpo_pair_loss(policy_logratio: f64, reference_logratio: f64, beta: f64) -> f64 {
    let negative_margin = -beta * (policy_logratio - reference_logratio);
    negative_margin.max(0.0) + (-negative_margin.abs()).exp().ln_1p()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_logratios_give_log_two() {
        assert!((dpo_pair_loss(0.4, 0.4, 0.5) - std::f64::consts::LN_2).abs() < 1e-12);
        assert!((dpo_pair_loss(0.8, 0.4, 0.5) - 0.5981388693815918).abs() < 1e-12);
        assert!(dpo_pair_loss(-1.0, 0.4, 0.5) > std::f64::consts::LN_2);
        assert!((dpo_pair_loss(-2000.0, 0.0, 0.5) - 1000.0).abs() < 1e-10);
    }
}
