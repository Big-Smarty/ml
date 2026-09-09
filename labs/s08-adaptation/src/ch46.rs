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

        // Baseline behavior cloning uses chosen labels and ignores rejected
        // odds after validation. LEARNER: implement stable relative DPO pairs,
        // accumulate their mean gradient at one old policy, and freeze reference.
        let mut loss = 0.;
        let mut gradient = Gradient {
            logits: self.logits.iter().map(|r| vec![0.; r.len()]).collect(),
        };
        for pair in data {
            let row = self.logits.get(pair.prompt).ok_or("invalid prompt")?;
            if pair.chosen >= row.len()
                || pair.rejected >= row.len()
                || pair.chosen == pair.rejected
            {
                return Err("invalid preference pair");
            }
            let logp = log_softmax(row)?;
            loss -= logp[pair.chosen] / data.len() as f64;
            for (i, p) in softmax(row)?.iter().enumerate() {
                gradient.logits[pair.prompt][i] +=
                    (p - f64::from(i == pair.chosen)) / data.len() as f64;
            }
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
        let _expected_reward: f64 = probabilities
            .iter()
            .zip(&rewards)
            .map(|(probability, reward)| probability * reward)
            .sum();
        for i in 0..logits.len() {
            let gradient = f64::from(rewards[i] == 1.0) - probabilities[i];
            logits[i] += learning_rate * gradient;
        }
    }
    softmax(&logits)
}

fn experiment() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Reference DPO pair at equal odds: {}",
        dpo_pair_loss(0., 0., 0.5)
    );
    let (policy, curve) = train_dpo(400, 0.2)?;
    println!(
        "chosen-only supervised baseline loss: {:.4} -> {:.4}",
        curve[0], curve[1]
    );
    println!(
        "learned prompt policies: {:?}",
        policy
            .logits
            .iter()
            .map(|row| softmax(row))
            .collect::<Result<Vec<_>, _>>()?
    );
    let verified = exact_reward_training(200, 0.5)?;
    println!("verifier-selected supervised target policy after 200 steps: {verified:.4?}");
    println!("The categorical fixture checks objective mechanics; it is not language-model alignment evidence.");
    Ok(())
}

/// Run the useful baseline; --check separately verifies the learning goal.
pub fn run(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("unexpected experiment argument".into());
    }
    experiment().map_err(|e| e.to_string())?;
    feedback_experiment().map_err(str::to_string)
}

/// Learn a scalar reward for each response feature vector from pairwise rankings.
/// Bradley-Terry: probability chosen wins = sigmoid(r_chosen-r_rejected).
fn train_reward_model(
    pairs: &[([f64; 2], [f64; 2])],
    steps: usize,
) -> Result<[f64; 2], &'static str> {
    if pairs.is_empty()
        || pairs
            .iter()
            .flat_map(|(a, b)| a.iter().chain(b))
            .any(|x| !x.is_finite())
    {
        return Err("invalid preference features");
    }
    // Baseline average feature difference ranks without fitting a probabilistic
    // reward model. LEARNER: train Bradley-Terry pair loss with mean gradients.
    let mut weights = [0.; 2];
    for (chosen, rejected) in pairs {
        for i in 0..2 {
            weights[i] += (chosen[i] - rejected[i]) / pairs.len() as f64;
        }
    }
    println!("feature-average baseline (requested {steps} optimizer steps; no optimizer yet)");
    Ok(weights)
}
fn feedback_experiment() -> Result<(), &'static str> {
    // Feature 0 is checked correctness, feature 1 is normalized response length.
    let pairs = [([1., 0.2], [0., 0.8]), ([1., 0.8], [0., 0.2])];
    let reward = train_reward_model(&pairs, 200)?;
    let held_correct = [1., 0.5];
    let held_wrong = [0., 0.5];
    let score = |x: [f64; 2]| reward[0] * x[0] + reward[1] * x[1];
    println!(
        "reward model weights={reward:.4?}; held-out equal-length correct/wrong scores {:.4}/{:.4}",
        score(held_correct),
        score(held_wrong)
    );
    let biased = train_reward_model(&[([1., 1.], [0., 0.])], 200)?;
    println!("confounded reward data: weights={biased:.4?}; novel wrong-but-long response [0,2] score={:.4}",2.*biased[1]);
    let reference = Policy {
        logits: vec![vec![0.; 3]],
    };
    let mut conflict = reference.clone();
    let pairs = [
        Preference {
            prompt: 0,
            chosen: 1,
            rejected: 0,
        },
        Preference {
            prompt: 0,
            chosen: 0,
            rejected: 1,
        },
    ];
    for _ in 0..100 {
        conflict.step(&reference, &pairs, 0.5, 0.2)?;
    }
    println!(
        "opposed-pair loss={:.6}; balanced conflicts cannot both reach zero",
        conflict.loss(&reference, &pairs, 0.5)?
    );
    println!(
        "incomplete verifier example: '5' passes 2+3 equality but supplies no requested derivation"
    );
    Ok(())
}
pub fn check() -> Result<(), String> {
    let verify = || -> Result<(), &'static str> {
        let reference = Policy {
            logits: vec![vec![0.1, -0.2, 0.]],
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
        let mut candidate = reference.clone();
        candidate.logits[0][0] += 0.4;
        let (_, g) = candidate.loss_and_gradient(&reference, &data, 0.7)?;
        let h = 1e-5;
        let mut plus = candidate.clone();
        plus.logits[0][0] += h;
        let mut minus = candidate.clone();
        minus.logits[0][0] -= h;
        let numeric =
            (plus.loss(&reference, &data, 0.7)? - minus.loss(&reference, &data, 0.7)?) / (2. * h);
        if (numeric - g.logits[0][0]).abs() > 1e-6 + 1e-4 * numeric.abs() {
            return Err("GOAL_NOT_MET: DPO batch gradient does not match its mean loss");
        }
        let expected = -0.7 / (1. + (0.7_f64 * 0.4).exp());
        if (g.logits[0][0] - expected).abs() > 1e-12 {
            return Err(
                "GOAL_NOT_MET: implement DPO's frozen-reference relative margin, not winner-only supervised loss",
            );
        }
        let (mut a, mut b) = (reference.clone(), reference.clone());
        a.step(&reference, &data, 0.7, 0.2)?;
        b.step(&reference, &[data[1], data[0]], 0.7, 0.2)?;
        if a.logits[0]
            .iter()
            .zip(&b.logits[0])
            .any(|(x, y)| (x - y).abs() > 1e-12)
        {
            return Err(
                "GOAL_NOT_MET: accumulate all pair gradients at the old policy before updating",
            );
        }
        let (policy, curve) = train_dpo(400, 0.2)?;
        if curve[1] >= 0.18 || policy.logits[0][1] <= policy.logits[0][0] {
            return Err("GOAL_NOT_MET: DPO must lower pair loss and rank chosen responses");
        }
        if exact_reward_training(200, 0.5)?[1] <= 0.99 {
            return Err(
                "GOAL_NOT_MET: verifier objective must optimize the probability-weighted reward",
            );
        }
        let first = exact_reward_training(1, 0.5)?;
        let expected = softmax(&[-1. / 18., 1. / 9., -1. / 18.])?;
        if first
            .iter()
            .zip(expected)
            .any(|(a, b)| (a - b).abs() > 1e-12)
        {
            return Err("GOAL_NOT_MET: implement expected reward gradient p_i*(r_i-J); choosing the best target optimizes a different objective");
        }
        let reward = train_reward_model(&[([1., 0.2], [0., 0.8]), ([1., 0.8], [0., 0.2])], 200)?;
        if reward[0] <= 1. || reward[1].abs() > 1e-10 {
            return Err("GOAL_NOT_MET: reward model must distinguish correctness from response length on counterbalanced pairs");
        }
        feedback_experiment()?;
        println!(
            "goal: DPO gradient/order, trained verifier policy and learned reward transfer pass"
        );
        Ok(())
    };
    verify().map_err(str::to_string)
}

#[cfg(test)]
mod baseline_tests {
    #[test]
    fn supplied_baseline_runs() {
        assert!(super::run(&[]).is_ok());
    }
}
