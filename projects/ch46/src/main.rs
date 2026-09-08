//! Direct preference optimization and exact policy gradients on tiny categorical policies.

fn softmax(logits: &[f64]) -> Result<Vec<f64>, &'static str> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err("softmax needs finite logits");
    }
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities: Vec<_> = logits.iter().map(|value| (value - maximum).exp()).collect();
    let sum: f64 = probabilities.iter().sum();
    probabilities.iter_mut().for_each(|value| *value /= sum);
    Ok(probabilities)
}

fn log_softmax(logits: &[f64]) -> Result<Vec<f64>, &'static str> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err("log-softmax needs finite logits");
    }
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_sum = logits
        .iter()
        .map(|value| (value - maximum).exp())
        .sum::<f64>()
        .ln();
    let result: Vec<_> = logits
        .iter()
        .map(|value| (value - maximum) - log_sum)
        .collect();
    if result.iter().any(|value| !value.is_finite()) {
        return Err("log-softmax range overflowed");
    }
    Ok(result)
}

#[derive(Clone, Copy)]
struct Preference {
    prompt: usize,
    chosen: usize,
    rejected: usize,
}

fn dpo_loss(
    policy: &[Vec<f64>],
    reference: &[Vec<f64>],
    pairs: &[Preference],
    beta: f64,
) -> Result<f64, &'static str> {
    if pairs.is_empty() || !beta.is_finite() || beta <= 0.0 || policy.len() != reference.len() {
        return Err("DPO needs aligned policies, pairs, and positive beta");
    }
    for (current, frozen) in policy.iter().zip(reference) {
        if current.is_empty()
            || current.len() != frozen.len()
            || current.iter().chain(frozen).any(|logit| !logit.is_finite())
        {
            return Err("all policy rows must have matching nonempty finite reference rows");
        }
    }
    let mut total = 0.0;
    for pair in pairs {
        let current = log_softmax(policy.get(pair.prompt).ok_or("prompt is out of range")?)?;
        let frozen = log_softmax(reference.get(pair.prompt).ok_or("prompt is out of range")?)?;
        if pair.chosen == pair.rejected
            || pair.chosen >= current.len()
            || pair.rejected >= current.len()
            || current.len() != frozen.len()
        {
            return Err("response is out of range");
        }
        let margin = beta
            * ((current[pair.chosen] - current[pair.rejected])
                - (frozen[pair.chosen] - frozen[pair.rejected]));
        let negative_margin = -margin;
        total += negative_margin.max(0.0) + (-negative_margin.abs()).exp().ln_1p();
    }
    let mean = total / pairs.len() as f64;
    if !mean.is_finite() {
        return Err("DPO loss became nonfinite");
    }
    Ok(mean)
}

fn dpo_step(
    policy: &mut [Vec<f64>],
    reference: &[Vec<f64>],
    pairs: &[Preference],
    beta: f64,
    rate: f64,
) -> Result<(), &'static str> {
    if !rate.is_finite() || rate <= 0.0 {
        return Err("learning rate must be finite and positive");
    }
    dpo_loss(policy, reference, pairs, beta)?;
    let mut gradients: Vec<Vec<f64>> = policy.iter().map(|row| vec![0.0; row.len()]).collect();
    for pair in pairs {
        let current = log_softmax(&policy[pair.prompt])?;
        let frozen = log_softmax(&reference[pair.prompt])?;
        let margin = beta
            * ((current[pair.chosen] - current[pair.rejected])
                - (frozen[pair.chosen] - frozen[pair.rejected]));
        let loss_slope = -beta / (1.0 + margin.exp()) / pairs.len() as f64;
        gradients[pair.prompt][pair.chosen] += loss_slope;
        gradients[pair.prompt][pair.rejected] -= loss_slope;
    }
    // Validate the complete candidate before mutation so an error preserves the old policy.
    for (row, gradient) in policy.iter().zip(&gradients) {
        for (logit, slope) in row.iter().zip(gradient) {
            if !(logit - rate * slope).is_finite() {
                return Err("DPO update produced a nonfinite logit");
            }
        }
    }
    for (row, gradient) in policy.iter_mut().zip(gradients) {
        for (logit, slope) in row.iter_mut().zip(gradient) {
            *logit -= rate * slope;
        }
    }
    Ok(())
}

fn train_dpo(steps: usize) -> Result<(Vec<Vec<f64>>, Vec<f64>), &'static str> {
    let reference = vec![vec![0.3, 0.1, -0.2], vec![0.0, 0.2, -0.1]];
    let pairs = [
        Preference {
            prompt: 0,
            chosen: 1,
            rejected: 0,
        },
        Preference {
            prompt: 0,
            chosen: 1,
            rejected: 2,
        },
        Preference {
            prompt: 1,
            chosen: 2,
            rejected: 0,
        },
        Preference {
            prompt: 1,
            chosen: 2,
            rejected: 1,
        },
    ];
    let mut policy = reference.clone();
    let mut curve = vec![dpo_loss(&policy, &reference, &pairs, 0.5)?];
    for _ in 0..steps {
        dpo_step(&mut policy, &reference, &pairs, 0.5, 0.2)?;
    }
    curve.push(dpo_loss(&policy, &reference, &pairs, 0.5)?);
    Ok((policy, curve))
}

fn arithmetic_reward(prompt: &str, candidate: i64) -> Result<f64, &'static str> {
    let (left, right) = prompt
        .split_once('+')
        .ok_or("prompt must contain one plus sign")?;
    let left: i64 = left.trim().parse().map_err(|_| "left operand is invalid")?;
    let right: i64 = right
        .trim()
        .parse()
        .map_err(|_| "right operand is invalid")?;
    let answer = left
        .checked_add(right)
        .ok_or("arithmetic sum overflows i64")?;
    Ok(f64::from(candidate == answer))
}

fn exact_reward_training(steps: usize) -> Result<Vec<f64>, &'static str> {
    let candidates = [4, 5, 6];
    let rewards: Vec<_> = candidates
        .iter()
        .map(|&candidate| arithmetic_reward("2 + 3", candidate))
        .collect::<Result<_, _>>()?;
    let mut logits = vec![0.0; rewards.len()];
    for _ in 0..steps {
        let probabilities = softmax(&logits)?;
        let expected_reward: f64 = probabilities
            .iter()
            .zip(&rewards)
            .map(|(probability, reward)| probability * reward)
            .sum();
        for i in 0..logits.len() {
            let gradient = probabilities[i] * (rewards[i] - expected_reward);
            logits[i] += 0.5 * gradient;
        }
    }
    softmax(&logits)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (policy, curve) = train_dpo(400)?;
    println!("DPO mean loss: {:.4} -> {:.4}", curve[0], curve[1]);
    println!(
        "learned prompt policies: {:?}",
        policy
            .iter()
            .map(|row| softmax(row))
            .collect::<Result<Vec<_>, _>>()?
    );
    let verified = exact_reward_training(200)?;
    println!("exact-reward policy after 200 steps: {verified:.4?}");
    println!("The categorical fixture checks objective mechanics; it is not language-model alignment evidence.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpo_lowers_loss_and_ranks_chosen_answers() -> Result<(), &'static str> {
        let (policy, curve) = train_dpo(400)?;
        assert!(curve[1] < curve[0] * 0.25);
        assert!(policy[0][1] > policy[0][0] && policy[0][1] > policy[0][2]);
        assert!(policy[1][2] > policy[1][0] && policy[1][2] > policy[1][1]);
        Ok(())
    }

    #[test]
    fn dpo_gradient_matches_central_difference() -> Result<(), &'static str> {
        let reference = vec![vec![0.1, -0.2, 0.0]];
        let pair = [
            Preference {
                prompt: 0,
                chosen: 0,
                rejected: 1,
            },
            Preference {
                prompt: 0,
                chosen: 0,
                rejected: 2,
            },
        ];
        let mut policy = reference.clone();
        let before = policy[0][0];
        dpo_step(&mut policy, &reference, &pair, 0.7, 1e-4)?;
        let analytic = (before - policy[0][0]) / 1e-4;
        let h = 1e-6;
        let mut plus = reference.clone();
        let mut minus = reference.clone();
        plus[0][0] += h;
        minus[0][0] -= h;
        let numerical = (dpo_loss(&plus, &reference, &pair, 0.7)?
            - dpo_loss(&minus, &reference, &pair, 0.7)?)
            / (2.0 * h);
        assert!((analytic - numerical).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn stable_probabilities_and_batch_order() -> Result<(), &'static str> {
        let log_p = log_softmax(&[1e16, 1e16])?;
        assert!((log_p[0] + std::f64::consts::LN_2).abs() < 1e-12);
        let reference = vec![vec![0.1, -0.2, 0.0]];
        let pairs = [
            Preference {
                prompt: 0,
                chosen: 0,
                rejected: 1,
            },
            Preference {
                prompt: 0,
                chosen: 0,
                rejected: 2,
            },
        ];
        let mut forward = reference.clone();
        let mut reverse = reference.clone();
        dpo_step(&mut forward, &reference, &pairs, 0.7, 0.1)?;
        dpo_step(&mut reverse, &reference, &[pairs[1], pairs[0]], 0.7, 0.1)?;
        for (a, b) in forward[0].iter().zip(&reverse[0]) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(reference, vec![vec![0.1, -0.2, 0.0]]);
        assert!(dpo_loss(&reference, &reference, &[], 0.7).is_err());
        assert!(dpo_loss(&reference, &reference, &pairs, f64::NAN).is_err());
        let mut invalid_unused = reference.clone();
        invalid_unused.push(vec![0.0]);
        let mut expected_unused = reference.clone();
        expected_unused.push(vec![0.0, 0.0]);
        assert!(dpo_loss(&invalid_unused, &expected_unused, &pairs, 0.7).is_err());
        let mut preserved = reference.clone();
        assert!(dpo_step(&mut preserved, &reference, &pairs, f64::MAX, f64::MAX).is_err());
        assert_eq!(preserved, reference);
        let bad = [Preference {
            prompt: 0,
            chosen: 3,
            rejected: 0,
        }];
        assert!(dpo_step(&mut forward, &reference, &bad, 0.7, 0.1).is_err());
        let extreme = vec![vec![-1000.0, 1000.0, 0.0]];
        assert!(dpo_loss(&extreme, &reference, &pairs, 0.7)?.is_finite());
        Ok(())
    }

    #[test]
    fn verifiable_reward_concentrates_on_correct_action() -> Result<(), &'static str> {
        assert_eq!(arithmetic_reward("2 + 3", 4)?, 0.0);
        assert_eq!(arithmetic_reward("2 + 3", 5)?, 1.0);
        assert!(arithmetic_reward("two + 3", 5).is_err());
        assert!(arithmetic_reward("9223372036854775807 + 1", 0).is_err());
        assert!(arithmetic_reward("-9223372036854775808 + -1", 0).is_err());
        assert_eq!(arithmetic_reward("-2 + 3", 1)?, 1.0);
        let probabilities = exact_reward_training(200)?;
        assert!(probabilities[1] > 0.99);
        Ok(())
    }
}
