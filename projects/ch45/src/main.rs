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
        let action = if rng.next_f64() < epsilon {
            rng.index(means.len())
        } else {
            argmax(&estimates)?
        };
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
    let mut q_values = [[0.0_f64; 2]; 5];
    let mut rng = Rng(seed);
    let learning_rate = 0.2;
    let discount_factor = 0.95;
    for episode in 0..episodes {
        let mut state = 0;
        let epsilon = 0.25 * (1.0 - episode as f64 / episodes as f64) + 0.02;
        for _ in 0..32 {
            let action = if rng.next_f64() < epsilon {
                Action::ALL[rng.index(2)]
            } else if q_values[state][1] >= q_values[state][0] {
                Action::Right
            } else {
                Action::Left
            };
            let (next, reward, done) = environment_step(state, action);
            let action_index = action.index();
            q_values[state][action_index] = q_update(
                q_values[state][action_index],
                reward,
                q_values[next][0].max(q_values[next][1]),
                done,
                learning_rate,
                discount_factor,
            );
            state = next;
            if done {
                break;
            }
        }
    }
    q_values
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let estimates = train_bandit(4_000, 0.1, 7)?;
    println!("epsilon-greedy value estimates: {estimates:.3?}");

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bandit_identifies_best_arm() -> Result<(), &'static str> {
        let estimates = train_bandit(8_000, 0.1, 7)?;
        assert_eq!(argmax(&estimates)?, 2);
        Ok(())
    }

    #[test]
    fn terminal_state_never_bootstraps_or_restarts() {
        assert!((q_update(0.2, 1.0, 99.0, true, 0.1, 0.95) - 0.28).abs() < 1e-12);
        assert!((q_update(0.2, 1.0, 0.5, false, 0.1, 0.9) - 0.325).abs() < 1e-12);
        assert_eq!(environment_step(3, Action::Right), (4, 1.0, true));
        assert_eq!(environment_step(4, Action::Left), (4, 0.0, true));
        assert_eq!(environment_step(4, Action::Right), (4, 0.0, true));
    }

    #[test]
    fn q_learning_finds_short_path() {
        let q = q_learning(600, 11);
        assert!(q[..4].iter().all(|values| values[1] > values[0]));
        let mut exact = 1.0;
        for state in (0..4).rev() {
            assert!((q[state][1] - exact).abs() < 1e-6);
            exact = -0.02 + 0.95 * exact;
        }
    }

    #[test]
    fn reinforce_changes_the_policy() -> Result<(), &'static str> {
        let probabilities = reinforce(5_000, 19)?;
        assert!(probabilities[1] > 0.9);
        Ok(())
    }

    #[test]
    fn invalid_bandit_configuration_is_rejected() {
        assert!(train_bandit(0, 0.1, 1).is_err());
        assert!(train_bandit(1, 1.1, 1).is_err());
    }
}
