//! Replace Chapter 45's sampled update with its exact two-action expectation, then complete DPO.

fn softmax2(logits: [f64; 2]) -> [f64; 2] {
    let maximum = logits[0].max(logits[1]);
    let values = [(logits[0] - maximum).exp(), (logits[1] - maximum).exp()];
    let sum = values[0] + values[1];
    [values[0] / sum, values[1] / sum]
}

fn reinforce_exact_step(logits: &mut [f64; 2], rewards: [f64; 2], learning_rate: f64) {
    let p = softmax2(*logits);
    let expected = p[0] * rewards[0] + p[1] * rewards[1];
    for i in 0..2 {
        logits[i] += learning_rate * p[i] * (rewards[i] - expected);
    }
}

fn dpo_pair_loss(policy_logratio: f64, reference_logratio: f64, beta: f64) -> f64 {
    // TODO: form the relative margin, then evaluate stable softplus of its negative.
    let _ = (policy_logratio, reference_logratio, beta);
    todo!("compute -ln(sigmoid(beta * (policy_logratio - reference_logratio)))")
}

fn main() {
    let _guided_todo: fn(f64, f64, f64) -> f64 = dpo_pair_loss;
    let mut logits = [0.0, 0.0];
    for _ in 0..100 {
        reinforce_exact_step(&mut logits, [0.0, 1.0], 0.5);
    }
    println!("exact two-action policy: {:.3?}", softmax2(logits));
    println!("Now run cargo test and implement dpo_pair_loss.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_policy_and_reference_logratios_have_log_two_loss() {
        assert!((dpo_pair_loss(0.3, 0.3, 0.5) - std::f64::consts::LN_2).abs() < 1e-12);
        assert!((dpo_pair_loss(0.8, 0.4, 0.5) - 0.5981388693815918).abs() < 1e-12);
        assert!((dpo_pair_loss(-2000.0, 0.0, 0.5) - 1000.0).abs() < 1e-10);
    }
}
