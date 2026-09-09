//! Bandits, tabular Q-learning, and REINFORCE without ML dependencies.

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn index(&mut self, count: usize) -> usize {
        (self.next_f64() * count as f64) as usize
    }
}

fn argmax(values: &[f64]) -> Result<usize, &'static str> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("argmax needs finite values");
    }
    Ok((1..values.len()).fold(0, |best, i| if values[i] > values[best] { i } else { best }))
}

fn train_bandit(steps: usize, epsilon: f64, seed: u64) -> Result<[f64; 3], &'static str> {
    if steps == 0 || !(0.0..=1.0).contains(&epsilon) {
        return Err("steps must be positive and epsilon must be in [0, 1]");
    }
    let means = [0.15, 0.55, 0.85];
    let mut estimates = [0.0; 3];
    let mut counts = [0_u64; 3];
    let mut rng = Rng(seed);
    for _ in 0..steps {
        // Working round-robin data collection; replace with epsilon-greedy selection.
        let action = counts.iter().sum::<u64>() as usize % means.len();
        let reward = if rng.next_f64() < means[action] {
            1.0
        } else {
            0.0
        };
        counts[action] += 1;
        estimates[action] += (reward - estimates[action]) / counts[action] as f64;
    }
    Ok(estimates)
}

#[derive(Clone, Copy)]
enum Action {
    Left,
    Right,
}

impl Action {
    const ALL: [Self; 2] = [Self::Left, Self::Right];

    fn index(self) -> usize {
        match self {
            Self::Left => 0,
            Self::Right => 1,
        }
    }
}

fn environment_step(state: usize, action: Action) -> (usize, f64, bool) {
    assert!(state <= 4, "closed chain state must be in 0..=4");
    if state == 4 {
        return (4, 0.0, true);
    }
    let next = match action {
        Action::Left => state.saturating_sub(1),
        Action::Right => (state + 1).min(4),
    };
    (next, if next == 4 { 1.0 } else { -0.02 }, next == 4)
}

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

fn q_learning(episodes: usize, seed: u64) -> [[f64; 2]; 5] {
    // Baseline estimates immediate reward under uniform exploration. This is
    // useful but cannot credit early actions for delayed terminal reward.
    // LEARNER: implement epsilon-greedy episodes, next-state TD targets and stop.
    let mut values = [[0.; 2]; 5];
    let mut counts = [[0; 2]; 5];
    let mut rng = Rng(seed);
    for _ in 0..episodes {
        let mut state = 0;
        for _ in 0..32 {
            let action = Action::ALL[rng.index(2)];
            let (next, reward, done) = environment_step(state, action);
            let a = action.index();
            counts[state][a] += 1;
            values[state][a] = q_update(
                values[state][a],
                reward,
                0.,
                true,
                1. / counts[state][a] as f64,
                0.95,
            );
            state = next;
            if done {
                break;
            }
        }
    }
    values
}

fn softmax2(logits: [f64; 2]) -> [f64; 2] {
    let maximum = logits[0].max(logits[1]);
    let values = [(logits[0] - maximum).exp(), (logits[1] - maximum).exp()];
    let sum = values[0] + values[1];
    [values[0] / sum, values[1] / sum]
}

fn reinforce(episodes: usize, seed: u64) -> Result<[f64; 2], &'static str> {
    if episodes == 0 {
        return Err("REINFORCE needs at least one episode");
    }
    let reward_probability = [0.2, 0.8];
    let mut logits = [0.0; 2];
    let mut baseline = 0.0;
    let mut rng = Rng(seed);
    let learning_rate = 0.08;
    let baseline_smoothing = 0.05;
    for _ in 0..episodes {
        let probabilities = softmax2(logits);
        let action = usize::from(rng.next_f64() >= probabilities[0]);
        let reward = f64::from(rng.next_f64() < reward_probability[action]);
        let advantage = reward - baseline;
        baseline += baseline_smoothing * advantage;
        for i in 0..2 {
            let grad_log_probability = f64::from(i == action) - probabilities[i];
            logits[i] += learning_rate * advantage * grad_log_probability;
        }
    }
    Ok(softmax2(logits))
}

fn experiment() -> Result<(), Box<dyn std::error::Error>> {
    let estimates = train_bandit(4_000, 0.1, 7)?;
    println!("round-robin baseline value estimates: {estimates:.3?}");

    let q = q_learning(600, 11);
    let policy: Vec<_> = q[..4]
        .iter()
        .map(|values| {
            if values[1] >= values[0] {
                "right"
            } else {
                "left"
            }
        })
        .collect();
    println!("grid policy from states 0..3: {policy:?}");

    let before = softmax2([0.0, 0.0]);
    let after = reinforce(5_000, 19)?;
    println!("REINFORCE action probabilities: {before:.3?} -> {after:.3?}");
    println!("Tiny simulated rewards verify mechanics, not performance in a real environment.");
    Ok(())
}

/// Run the useful baseline; --check separately verifies the learning goal.
pub fn run(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("unexpected experiment argument".into());
    }
    experiment().map_err(|e| e.to_string())?;
    controlled_experiment().map_err(str::to_string)
}

/// PPO's clipped surrogate is evaluated using probabilities from the frozen collection policy.
fn ppo_surrogate(
    probability: f64,
    old_probability: f64,
    advantage: f64,
    clip: f64,
) -> Result<(f64, f64), &'static str> {
    if !(0.0..=1.0).contains(&probability)
        || !(0.0..=1.0).contains(&old_probability)
        || old_probability == 0.
        || !advantage.is_finite()
        || !(0.0..1.0).contains(&clip)
    {
        return Err("PPO needs valid probabilities, finite advantage and clip in (0,1)");
    }
    let ratio = probability / old_probability;
    let raw = ratio * advantage;
    let clipped = ratio.clamp(1. - clip, 1. + clip) * advantage;
    let active = raw <= clipped;
    Ok((
        raw.min(clipped),
        if active {
            advantage / old_probability
        } else {
            0.
        },
    ))
}
fn ppo_train(seed: u64) -> Result<[f64; 2], &'static str> {
    // Working on-policy REINFORCE baseline. Build batched PPO: collect actions
    // and advantages, freeze old probabilities, reuse batch with clipped slopes.
    // Run the scalar clipping examples before changing the training loop.
    let _trace = ppo_surrogate(0.8, 0.5, 1., 0.2)?;
    reinforce(100, seed)
}
fn bandit_regret(steps: usize, epsilon: f64, seed: u64) -> Result<f64, &'static str> {
    if steps == 0 || !(0.0..=1.0).contains(&epsilon) {
        return Err("invalid bandit configuration");
    }
    let means = [0.15, 0.55, 0.85];
    let mut values = [0.; 3];
    let mut counts = [0; 3];
    let mut rng = Rng(seed);
    let mut regret = 0.;
    for _ in 0..steps {
        let action = if rng.next_f64() < epsilon {
            rng.index(3)
        } else {
            argmax(&values)?
        };
        let reward = f64::from(rng.next_f64() < means[action]);
        counts[action] += 1;
        values[action] += (reward - values[action]) / counts[action] as f64;
        regret += 0.85 - means[action];
    }
    Ok(regret)
}
fn controlled_experiment() -> Result<(), &'static str> {
    for epsilon in [0., 0.1, 0.3] {
        let mut regrets = Vec::new();
        for seed in [7, 19, 31, 47, 59] {
            regrets.push(bandit_regret(4000, epsilon, seed)?);
        }
        println!(
            "epsilon={epsilon}, expected cumulative regret seeds 7,19,31,47,59: {regrets:.1?}"
        );
    }
    println!(
        "on-policy 100-sample baseline before PPO: {:?}",
        ppo_train(45)?
    );
    Ok(())
}
pub fn check() -> Result<(), String> {
    let verify = || -> Result<(), &'static str> {
        let estimates = train_bandit(8000, 0.1, 7)?;
        if argmax(&estimates)? != 2 {
            return Err("GOAL_NOT_MET: implement action-specific bandit estimates and exploration");
        }
        let greedy_only = train_bandit(100, 0., 7)?;
        if greedy_only[1..].iter().any(|value| *value != 0.) {
            return Err("GOAL_NOT_MET: epsilon zero must select only the initial greedy arm in this fixture; replace round-robin collection");
        }
        let q = q_learning(600, 11);
        let mut exact = 1.;
        for state in (0..4).rev() {
            if (q[state][1] - exact).abs() > 1e-6 || q[state][1] <= q[state][0] {
                return Err("GOAL_NOT_MET: implement complete Q-learning interaction and terminal bootstrap; right values must match exact path returns");
            }
            exact = -0.02 + 0.95 * exact;
        }
        if reinforce(5000, 19)?[1] < 0.9 {
            return Err(
                "GOAL_NOT_MET: implement sampled REINFORCE using old probabilities and the pre-update baseline",
            );
        }
        if ppo_surrogate(0.8, 0.5, 1., 0.2)? != (1.2, 0.)
            || ppo_surrogate(0.2, 0.5, -1., 0.2)? != (-0.8, 0.)
        {
            return Err(
                "GOAL_NOT_MET: PPO clipping must stop improvement beyond the bound for both advantage signs",
            );
        }
        if ppo_train(45)?[1] < 0.85 {
            return Err(
                "GOAL_NOT_MET: PPO must train with frozen old-policy probabilities across each batch's epochs",
            );
        }
        controlled_experiment()?;
        println!("goal: bandit, exact chain values, REINFORCE and PPO training pass");
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
