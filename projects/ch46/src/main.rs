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

fn dpo_pair_loss(policy_logratio: f64, reference_logratio: f64, beta: f64) -> f64 {
    let negative_margin = -beta * (policy_logratio - reference_logratio);
    negative_margin.max(0.0) + (-negative_margin.abs()).exp().ln_1p()
}

#[derive(Clone, Debug, PartialEq)]
struct Policy {
    logits: Vec<Vec<f64>>,
}

#[derive(Debug)]
struct Gradient {
    logits: Vec<Vec<f64>>,
}

impl Policy {
    fn loss_and_gradient(
        &self,
        reference: &Self,
        data: &[Preference],
        beta: f64,
    ) -> Result<(f64, Gradient), &'static str> {
        if data.is_empty()
            || !beta.is_finite()
            || beta <= 0.0
            || self.logits.len() != reference.logits.len()
        {
            return Err("DPO needs aligned policies, data, and positive beta");
        }
        for (policy_row, reference_row) in self.logits.iter().zip(&reference.logits) {
            if policy_row.is_empty()
                || policy_row.len() != reference_row.len()
                || policy_row
                    .iter()
                    .chain(reference_row)
                    .any(|logit| !logit.is_finite())
            {
                return Err("all policy rows must have matching nonempty finite reference rows");
            }
        }

        let mut total_loss = 0.0;
        let mut gradient = Gradient {
            logits: self.logits.iter().map(|row| vec![0.0; row.len()]).collect(),
        };
        for pair in data {
            let policy_log_probabilities = log_softmax(
                self.logits
                    .get(pair.prompt)
                    .ok_or("prompt is out of range")?,
            )?;
            let reference_log_probabilities = log_softmax(
                reference
                    .logits
                    .get(pair.prompt)
                    .ok_or("prompt is out of range")?,
            )?;
            if pair.chosen == pair.rejected
                || pair.chosen >= policy_log_probabilities.len()
                || pair.rejected >= policy_log_probabilities.len()
            {
                return Err("response is out of range");
            }
            let policy_logratio =
                policy_log_probabilities[pair.chosen] - policy_log_probabilities[pair.rejected];
            let reference_logratio = reference_log_probabilities[pair.chosen]
                - reference_log_probabilities[pair.rejected];
            let margin = beta * (policy_logratio - reference_logratio);
            total_loss += dpo_pair_loss(policy_logratio, reference_logratio, beta);
            let loss_slope = -beta / (1.0 + margin.exp()) / data.len() as f64;
            gradient.logits[pair.prompt][pair.chosen] += loss_slope;
            gradient.logits[pair.prompt][pair.rejected] -= loss_slope;
        }
        let loss = total_loss / data.len() as f64;
        if !loss.is_finite()
            || gradient
                .logits
                .iter()
                .flatten()
                .any(|slope| !slope.is_finite())
        {
            return Err("DPO loss or gradient became nonfinite");
        }
        Ok((loss, gradient))
    }

    fn loss(&self, reference: &Self, data: &[Preference], beta: f64) -> Result<f64, &'static str> {
        Ok(self.loss_and_gradient(reference, data, beta)?.0)
    }

    fn step(
        &mut self,
        reference: &Self,
        data: &[Preference],
        beta: f64,
        learning_rate: f64,
    ) -> Result<(), &'static str> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        let (_, gradient) = self.loss_and_gradient(reference, data, beta)?;
        // Validate the complete candidate before mutation so an error preserves the old policy.
        for (row, gradient_row) in self.logits.iter().zip(&gradient.logits) {
            for (logit, slope) in row.iter().zip(gradient_row) {
                if !(logit - learning_rate * slope).is_finite() {
                    return Err("DPO update produced a nonfinite logit");
                }
            }
        }
        for (row, gradient_row) in self.logits.iter_mut().zip(gradient.logits) {
            for (logit, slope) in row.iter_mut().zip(gradient_row) {
                *logit -= learning_rate * slope;
            }
        }
        Ok(())
    }
}

fn train_dpo(steps: usize, learning_rate: f64) -> Result<(Policy, [f64; 2]), &'static str> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("learning rate must be finite and positive");
    }
    let reference = Policy {
        logits: vec![vec![0.3, 0.1, -0.2], vec![0.0, 0.2, -0.1]],
    };
    let data = [
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
    let before = policy.loss(&reference, &data, 0.5)?;
    for _ in 0..steps {
        policy.step(&reference, &data, 0.5, learning_rate)?;
    }
    let after = policy.loss(&reference, &data, 0.5)?;
    Ok((policy, [before, after]))
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

fn exact_reward_training(steps: usize, learning_rate: f64) -> Result<Vec<f64>, &'static str> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("exact-reward training needs a positive learning rate");
    }
    let prompt = "2 + 3";
    let candidates = [4, 5, 6];
    let rewards: Vec<_> = candidates
        .iter()
        .map(|&candidate| arithmetic_reward(prompt, candidate))
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
            logits[i] += learning_rate * gradient;
        }
    }
    softmax(&logits)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (policy, curve) = train_dpo(400, 0.2)?;
    println!("DPO mean loss: {:.4} -> {:.4}", curve[0], curve[1]);
    println!(
        "learned prompt policies: {:?}",
        policy
            .logits
            .iter()
            .map(|row| softmax(row))
            .collect::<Result<Vec<_>, _>>()?
    );
    let verified = exact_reward_training(200, 0.5)?;
    println!("exact-reward policy after 200 steps: {verified:.4?}");
    println!("The categorical fixture checks objective mechanics; it is not language-model alignment evidence.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpo_lowers_loss_and_ranks_chosen_answers() -> Result<(), &'static str> {
        let (policy, curve) = train_dpo(400, 0.2)?;
        assert!(curve[1] < curve[0] * 0.25);
        assert!(
            policy.logits[0][1] > policy.logits[0][0] && policy.logits[0][1] > policy.logits[0][2]
        );
        assert!(
            policy.logits[1][2] > policy.logits[1][0] && policy.logits[1][2] > policy.logits[1][1]
        );
        Ok(())
    }

    #[test]
    fn dpo_gradient_matches_central_difference() -> Result<(), &'static str> {
        let reference = Policy {
            logits: vec![vec![0.1, -0.2, 0.0]],
        };
        let data = [
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
        let before = policy.logits[0][0];
        policy.step(&reference, &data, 0.7, 1e-4)?;
        let analytic = (before - policy.logits[0][0]) / 1e-4;
        let h = 1e-6;
        let mut plus = reference.clone();
        let mut minus = reference.clone();
        plus.logits[0][0] += h;
        minus.logits[0][0] -= h;
        let numerical =
            (plus.loss(&reference, &data, 0.7)? - minus.loss(&reference, &data, 0.7)?) / (2.0 * h);
        assert!((analytic - numerical).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn stable_probabilities_and_batch_order() -> Result<(), &'static str> {
        let log_p = log_softmax(&[1e16, 1e16])?;
        assert!((log_p[0] + std::f64::consts::LN_2).abs() < 1e-12);
        assert!((dpo_pair_loss(0.3, 0.3, 0.5) - std::f64::consts::LN_2).abs() < 1e-12);
        assert!((dpo_pair_loss(-2000.0, 0.0, 0.5) - 1000.0).abs() < 1e-10);
        let reference = Policy {
            logits: vec![vec![0.1, -0.2, 0.0]],
        };
        let data = [
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
        forward.step(&reference, &data, 0.7, 0.1)?;
        reverse.step(&reference, &[data[1], data[0]], 0.7, 0.1)?;
        for (a, b) in forward.logits[0].iter().zip(&reverse.logits[0]) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(reference.logits, vec![vec![0.1, -0.2, 0.0]]);
        assert!(reference.loss(&reference, &[], 0.7).is_err());
        assert!(reference.loss(&reference, &data, f64::NAN).is_err());
        let mut invalid_unused = reference.clone();
        invalid_unused.logits.push(vec![0.0]);
        let mut expected_unused = reference.clone();
        expected_unused.logits.push(vec![0.0, 0.0]);
        assert!(invalid_unused.loss(&expected_unused, &data, 0.7).is_err());
        let mut preserved = reference.clone();
        assert!(preserved
            .step(&reference, &data, f64::MAX, f64::MAX)
            .is_err());
        assert_eq!(preserved, reference);
        let bad = [Preference {
            prompt: 0,
            chosen: 3,
            rejected: 0,
        }];
        assert!(forward.step(&reference, &bad, 0.7, 0.1).is_err());
        let extreme = Policy {
            logits: vec![vec![-1000.0, 1000.0, 0.0]],
        };
        assert!(extreme.loss(&reference, &data, 0.7)?.is_finite());
        assert!(train_dpo(0, f64::NAN).is_err());
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
        let probabilities = exact_reward_training(200, 0.5)?;
        assert!(probabilities[1] > 0.99);
        assert!(exact_reward_training(0, f64::NAN).is_err());
        Ok(())
    }
}
