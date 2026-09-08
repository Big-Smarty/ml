//! A trainable top-1 mixture-of-experts layer with capacity and load balancing.

const INPUTS: usize = 2;
const EXPERTS: usize = 3;

#[derive(Clone, Debug)]
struct Moe {
    router: [[f64; INPUTS]; EXPERTS],
    expert_weight: [[f64; INPUTS]; EXPERTS],
    expert_bias: [f64; EXPERTS],
}

impl Moe {
    fn new() -> Self {
        Self {
            router: [[0.0, 0.0], [0.01, -0.02], [-0.02, 0.01]],
            expert_weight: [[0.1, -0.1], [-0.1, 0.05], [0.05, 0.1]],
            expert_bias: [0.0; EXPERTS],
        }
    }

    fn probabilities(&self, input: [f64; INPUTS]) -> [f64; EXPERTS] {
        let logits = self.router.map(|row| row[0] * input[0] + row[1] * input[1]);
        let maximum = logits.into_iter().fold(f64::NEG_INFINITY, f64::max);
        let mut probabilities = logits.map(|logit| (logit - maximum).exp());
        let sum: f64 = probabilities.iter().sum();
        probabilities.iter_mut().for_each(|value| *value /= sum);
        probabilities
    }

    fn route(&self, input: [f64; INPUTS]) -> (usize, [f64; EXPERTS]) {
        let probabilities = self.probabilities(input);
        let selected = (1..EXPERTS).fold(0, |best, i| {
            if probabilities[i] > probabilities[best] {
                i
            } else {
                best
            }
        });
        (selected, probabilities)
    }

    fn expert(&self, index: usize, input: [f64; INPUTS]) -> f64 {
        self.expert_weight[index][0] * input[0]
            + self.expert_weight[index][1] * input[1]
            + self.expert_bias[index]
    }

    fn predict(&self, input: [f64; INPUTS]) -> f64 {
        let (selected, probabilities) = self.route(input);
        probabilities[selected] * self.expert(selected, input)
    }

    fn mean_loss(&self, data: &[([f64; INPUTS], f64)]) -> Result<f64, &'static str> {
        let model_finite = self
            .router
            .iter()
            .chain(self.expert_weight.iter())
            .flatten()
            .chain(self.expert_bias.iter())
            .all(|value| value.is_finite());
        if !model_finite
            || data.is_empty()
            || data.iter().any(|(input, target)| {
                !target.is_finite() || input.iter().any(|value| !value.is_finite())
            })
        {
            return Err("loss needs finite examples");
        }
        let loss = data
            .iter()
            .map(|&(input, target)| 0.5 * (self.predict(input) - target).powi(2))
            .sum::<f64>()
            / data.len() as f64;
        if !loss.is_finite() {
            return Err("loss became nonfinite");
        }
        Ok(loss)
    }

    fn train_epoch(
        &mut self,
        data: &[([f64; INPUTS], f64)],
        rate: f64,
        capacity_factor: f64,
        balance_weight: f64,
        offset: usize,
    ) -> Result<usize, &'static str> {
        if data.is_empty()
            || !rate.is_finite()
            || rate <= 0.0
            || !capacity_factor.is_finite()
            || capacity_factor <= 0.0
            || !balance_weight.is_finite()
            || balance_weight < 0.0
        {
            return Err("training configuration must be finite and positive");
        }
        self.mean_loss(data)?;
        let capacity = ((data.len() as f64 / EXPERTS as f64) * capacity_factor)
            .ceil()
            .max(1.0) as usize;
        let routes: Vec<_> = data.iter().map(|&(input, _)| self.route(input)).collect();
        let mut hard_counts = [0_usize; EXPERTS];
        routes
            .iter()
            .for_each(|(expert, _)| hard_counts[*expert] += 1);
        let frequencies = hard_counts.map(|count| count as f64 / data.len() as f64);

        let mut router_grad = [[0.0; INPUTS]; EXPERTS];
        let mut expert_grad = [[0.0; INPUTS]; EXPERTS];
        let mut bias_grad = [0.0; EXPERTS];
        let mut used = [0_usize; EXPERTS];
        let mut dropped = 0;

        for step in 0..data.len() {
            let i = step
                .checked_add(offset % data.len())
                .ok_or("example offset overflowed")?
                % data.len();
            let (input, target) = data[i];
            let (selected, probabilities) = routes[i];
            let mut probability_grad = frequencies
                .map(|frequency| balance_weight * EXPERTS as f64 * frequency / data.len() as f64);
            if used[selected] < capacity {
                used[selected] += 1;
                let expert_output = self.expert(selected, input);
                let error = probabilities[selected] * expert_output - target;
                let scale = error * probabilities[selected] / data.len() as f64;
                for feature in 0..INPUTS {
                    expert_grad[selected][feature] += scale * input[feature];
                }
                bias_grad[selected] += scale;
                probability_grad[selected] += error * expert_output / data.len() as f64;
            } else {
                dropped += 1;
            }

            let weighted: f64 = probability_grad
                .iter()
                .zip(probabilities)
                .map(|(gradient, probability)| gradient * probability)
                .sum();
            for expert in 0..EXPERTS {
                let logit_grad = probabilities[expert] * (probability_grad[expert] - weighted);
                for feature in 0..INPUTS {
                    router_grad[expert][feature] += logit_grad * input[feature];
                }
            }
        }

        for expert in 0..EXPERTS {
            for feature in 0..INPUTS {
                self.router[expert][feature] -= rate * router_grad[expert][feature];
                self.expert_weight[expert][feature] -= rate * expert_grad[expert][feature];
            }
            self.expert_bias[expert] -= rate * bias_grad[expert];
        }
        self.mean_loss(data)?;
        Ok(dropped)
    }
}

fn fixture() -> Vec<([f64; INPUTS], f64)> {
    let mut data = Vec::new();
    for i in 0..18 {
        let t = 0.5 + i as f64 * 0.08;
        data.push(([t, 0.2], 1.5 * t + 0.4));
        data.push(([-t, 0.3], -0.8 * t - 0.7));
        data.push(([0.1, t], 0.5 * t - 0.2));
    }
    data
}

fn train(epochs: usize) -> Result<(Moe, f64, f64, usize), &'static str> {
    let data = fixture();
    let mut model = Moe::new();
    let before = model.mean_loss(&data)?;
    let mut dropped = 0;
    for epoch in 0..epochs {
        dropped = model.train_epoch(&data, 0.08, 1.25, 0.02, epoch % data.len())?;
    }
    let after = model.mean_loss(&data)?;
    Ok((model, before, after, dropped))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (model, before, after, dropped) = train(1_500)?;
    let mut counts = [0; EXPERTS];
    fixture()
        .iter()
        .for_each(|&(input, _)| counts[model.route(input).0] += 1);
    println!("MoE mean half-squared loss: {before:.5} -> {after:.5}");
    println!("route counts={counts:?}, final-epoch overflow={dropped}");
    println!(
        "router={:?}\nexperts={:?}",
        model.router, model.expert_weight
    );
    println!("This tiny CPU layer verifies routing gradients and capacity, not large-model quality or speed.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_full_softmax_gate_has_router_gradient() -> Result<(), &'static str> {
        let example = [([1.0, 0.2], 1.4)];
        let mut model = Moe::new();
        let selected = model.route(example[0].0).0;
        let before_router = model.router[selected][0];
        let before_expert = model.expert_weight[selected][0];
        let rate = 1e-5;
        model.train_epoch(&example, rate, 3.0, 0.0, 0)?;
        let analytic_router = (before_router - model.router[selected][0]) / rate;
        let analytic_expert = (before_expert - model.expert_weight[selected][0]) / rate;
        let h = 1e-6;
        let mut plus = Moe::new();
        let mut minus = Moe::new();
        plus.router[selected][0] += h;
        minus.router[selected][0] -= h;
        let numerical_router = (plus.mean_loss(&example)? - minus.mean_loss(&example)?) / (2.0 * h);
        let mut plus = Moe::new();
        let mut minus = Moe::new();
        plus.expert_weight[selected][0] += h;
        minus.expert_weight[selected][0] -= h;
        let numerical_expert = (plus.mean_loss(&example)? - minus.mean_loss(&example)?) / (2.0 * h);
        assert!((analytic_router - numerical_router).abs() < 1e-6);
        assert!((analytic_expert - numerical_expert).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn router_and_experts_train_and_loss_falls() -> Result<(), &'static str> {
        let initial = Moe::new();
        let (model, before, after, _) = train(1_500)?;
        assert!(after < before * 0.2, "{before} -> {after}");
        assert_ne!(model.router, initial.router);
        assert_ne!(model.expert_weight, initial.expert_weight);
        Ok(())
    }

    #[test]
    fn balance_gradient_matches_fixed_frequency_objective() -> Result<(), &'static str> {
        let data = fixture();
        let initial = Moe::new();
        let mut counts = [0.0; EXPERTS];
        for &(x, _) in &data {
            counts[initial.route(x).0] += 1.0;
        }
        let alpha = 0.02;
        let objective = |model: &Moe| -> Result<f64, &'static str> {
            let auxiliary: f64 = data
                .iter()
                .map(|&(x, _)| {
                    model
                        .probabilities(x)
                        .iter()
                        .zip(counts)
                        .map(|(p, count)| p * count / data.len() as f64)
                        .sum::<f64>()
                })
                .sum::<f64>()
                * alpha
                * EXPERTS as f64
                / data.len() as f64;
            Ok(model.mean_loss(&data)? + auxiliary)
        };
        let rate = 1e-5;
        let mut updated = initial.clone();
        assert_eq!(updated.train_epoch(&data, rate, 3.0, alpha, 0)?, 0);
        for expert in 0..EXPERTS {
            for feature in 0..INPUTS {
                let mut plus = initial.clone();
                let mut minus = initial.clone();
                plus.router[expert][feature] += 1e-5;
                minus.router[expert][feature] -= 1e-5;
                let numerical = (objective(&plus)? - objective(&minus)?) / 2e-5;
                let analytic =
                    (initial.router[expert][feature] - updated.router[expert][feature]) / rate;
                assert!((analytic - numerical).abs() < 1e-6 + 1e-4 * numerical.abs());
            }
        }
        Ok(())
    }

    #[test]
    fn capacity_reports_overflow() -> Result<(), &'static str> {
        let mut model = Moe::new();
        let data = fixture();
        let capacity = 9;
        let mut counts = [0_usize; EXPERTS];
        data.iter()
            .for_each(|&(input, _)| counts[model.route(input).0] += 1);
        let expected: usize = counts
            .iter()
            .map(|&count| count.saturating_sub(capacity))
            .sum();
        let dropped = model.train_epoch(&data, 0.01, 0.5, 0.0, 0)?;
        assert_eq!(dropped, expected);
        assert!(expected > 0);
        Ok(())
    }
}
