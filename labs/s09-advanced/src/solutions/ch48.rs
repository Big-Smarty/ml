//! Worked solution: Freeze the old router/expert state for the batch. Count all attempted routes for the auxiliary objective but admit only capacity-bounded expert work. A selected gate retains its full-softmax probability, so task and balance gradients both flow through every router logit. Apply accumulated gradients only after processing the batch.
//! Compare intermediate quantities with the lesson before using the final goal check.
//! Worked solution for chapter 48. Read the comments and lesson explanations before comparing.
include!("../common/ch48.rs");
include!("../checks/ch48.rs");
impl Model {
    fn probabilities(&self, input: [f64; INPUTS]) -> [f64; EXPERTS] {
        let logits = self
            .router_weights
            .map(|weights| weights[0] * input[0] + weights[1] * input[1]);
        softmax(logits)
    }
}

impl Model {
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
}

impl Model {
    fn train_epoch(
        &mut self,
        data: &[([f64; INPUTS], f64)],
        learning_rate: f64,
        capacity_factor: f64,
        balance_weight: f64,
        example_offset: usize,
    ) -> Result<usize, &'static str> {
        if data.is_empty()
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
            || !capacity_factor.is_finite()
            || capacity_factor <= 0.0
            || !balance_weight.is_finite()
            || balance_weight < 0.0
        {
            return Err("training configuration must be finite and positive");
        }
        self.loss(data)?;
        let capacity = ((data.len() as f64 / EXPERTS as f64) * capacity_factor)
            .ceil()
            .max(1.0) as usize;
        let routes: Vec<_> = data.iter().map(|&(input, _)| self.route(input)).collect();
        let mut attempted_counts = [0_usize; EXPERTS];
        routes
            .iter()
            .for_each(|(expert, _)| attempted_counts[*expert] += 1);
        let attempted_frequencies = attempted_counts.map(|count| count as f64 / data.len() as f64);

        let mut router_weight_gradient = [[0.0; INPUTS]; EXPERTS];
        let mut expert_weight_gradient = [[0.0; INPUTS]; EXPERTS];
        let mut expert_bias_gradient = [0.0; EXPERTS];
        let mut used = [0_usize; EXPERTS];
        let mut dropped = 0;

        for batch_position in 0..data.len() {
            let i = batch_position
                .checked_add(example_offset % data.len())
                .ok_or("example offset overflowed")?
                % data.len();
            let (input, target) = data[i];
            let (selected, probabilities) = routes[i];
            let mut probability_gradient = attempted_frequencies
                .map(|frequency| balance_weight * EXPERTS as f64 * frequency / data.len() as f64);
            if used[selected] < capacity {
                used[selected] += 1;
                let expert_output = self.expert(selected, input);
                let error = probabilities[selected] * expert_output - target;
                let scale = error * probabilities[selected] / data.len() as f64;
                for feature in 0..INPUTS {
                    expert_weight_gradient[selected][feature] += scale * input[feature];
                }
                expert_bias_gradient[selected] += scale;
                probability_gradient[selected] += error * expert_output / data.len() as f64;
            } else {
                dropped += 1;
            }

            let weighted: f64 = probability_gradient
                .iter()
                .zip(probabilities)
                .map(|(gradient, probability)| gradient * probability)
                .sum();
            for expert in 0..EXPERTS {
                let logit_gradient =
                    probabilities[expert] * (probability_gradient[expert] - weighted);
                for feature in 0..INPUTS {
                    router_weight_gradient[expert][feature] += logit_gradient * input[feature];
                }
            }
        }

        for expert in 0..EXPERTS {
            for feature in 0..INPUTS {
                self.router_weights[expert][feature] -=
                    learning_rate * router_weight_gradient[expert][feature];
                self.expert_weights[expert][feature] -=
                    learning_rate * expert_weight_gradient[expert][feature];
            }
            self.expert_biases[expert] -= learning_rate * expert_bias_gradient[expert];
        }
        self.loss(data)?;
        Ok(dropped)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_full_softmax_gate_has_router_gradient() -> Result<(), &'static str> {
        let example = [([1.0, 0.2], 1.4)];
        let mut model = Model::new();
        let selected = model.route(example[0].0).0;
        let before_router = model.router_weights[selected][0];
        let before_expert = model.expert_weights[selected][0];
        let learning_rate = 1e-5;
        model.train_epoch(&example, learning_rate, 3.0, 0.0, 0)?;
        let analytic_router = (before_router - model.router_weights[selected][0]) / learning_rate;
        let analytic_expert = (before_expert - model.expert_weights[selected][0]) / learning_rate;
        let h = 1e-6;
        let mut plus = Model::new();
        let mut minus = Model::new();
        plus.router_weights[selected][0] += h;
        minus.router_weights[selected][0] -= h;
        let numerical_router = (plus.loss(&example)? - minus.loss(&example)?) / (2.0 * h);
        let mut plus = Model::new();
        let mut minus = Model::new();
        plus.expert_weights[selected][0] += h;
        minus.expert_weights[selected][0] -= h;
        let numerical_expert = (plus.loss(&example)? - minus.loss(&example)?) / (2.0 * h);
        assert!((analytic_router - numerical_router).abs() < 1e-6);
        assert!((analytic_expert - numerical_expert).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn router_and_experts_train_and_loss_falls() -> Result<(), &'static str> {
        let initial = Model::new();
        let (model, before, after, _) = train_fixture(1_500)?;
        assert!(after < before * 0.2, "{before} -> {after}");
        assert_ne!(model.router_weights, initial.router_weights);
        assert_ne!(model.expert_weights, initial.expert_weights);
        Ok(())
    }

    #[test]
    fn balance_gradient_matches_fixed_frequency_objective() -> Result<(), &'static str> {
        let data = fixture();
        let initial = Model::new();
        let mut counts = [0.0; EXPERTS];
        for &(x, _) in &data {
            counts[initial.route(x).0] += 1.0;
        }
        let alpha = 0.02;
        let objective = |model: &Model| -> Result<f64, &'static str> {
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
            Ok(model.loss(&data)? + auxiliary)
        };
        let learning_rate = 1e-5;
        let mut updated = initial.clone();
        assert_eq!(updated.train_epoch(&data, learning_rate, 3.0, alpha, 0)?, 0);
        for expert in 0..EXPERTS {
            for feature in 0..INPUTS {
                let mut plus = initial.clone();
                let mut minus = initial.clone();
                plus.router_weights[expert][feature] += 1e-5;
                minus.router_weights[expert][feature] -= 1e-5;
                let numerical = (objective(&plus)? - objective(&minus)?) / 2e-5;
                let analytic = (initial.router_weights[expert][feature]
                    - updated.router_weights[expert][feature])
                    / learning_rate;
                assert!((analytic - numerical).abs() < 1e-6 + 1e-4 * numerical.abs());
            }
        }
        Ok(())
    }

    #[test]
    fn capacity_reports_overflow() -> Result<(), &'static str> {
        let mut model = Model::new();
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

    #[test]
    fn overflow_task_gradient_uses_all_examples_as_denominator() -> Result<(), &'static str> {
        let data = [([1.0, 0.0], 1.4), ([1.0, 0.0], 1.4)];
        let mut model = Model::new();
        let selected = model.route(data[0].0).0;
        let probability = model.probabilities(data[0].0)[selected];
        let error = model.predict(data[0].0) - data[0].1;
        let before = model.expert_weights[selected][0];
        let learning_rate = 1e-5;

        assert_eq!(model.train_epoch(&data, learning_rate, 0.1, 0.0, 0)?, 1);
        let gradient = (before - model.expert_weights[selected][0]) / learning_rate;
        assert!((gradient - error * probability / data.len() as f64).abs() < 1e-11);
        Ok(())
    }
}
