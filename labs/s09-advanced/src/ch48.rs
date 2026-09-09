//! Chapter 48 learner algorithms. Baseline trains experts with a frozen router and no admission limit. Implement the complete routed update: old-state routes, capacity, expert gradients, selected-gate router gradients, and stop-gradient hard-load balancing.
include!("common/ch48.rs");
include!("checks/ch48.rs");
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
        // Fixed-router regression: no capacity or auxiliary loss yet.
        let _ = (capacity_factor, balance_weight, example_offset);
        let mut dw = [[0.0; INPUTS]; EXPERTS];
        let mut db = [0.0; EXPERTS];
        for &(input, target) in data {
            let (selected, p) = self.route(input);
            let error = self.predict(input) - target;
            let scale = error * p[selected] / data.len() as f64;
            for (g, x) in dw[selected].iter_mut().zip(input) {
                *g += scale * x;
            }
            db[selected] += scale;
        }
        for expert in 0..EXPERTS {
            for (w, g) in self.expert_weights[expert].iter_mut().zip(dw[expert]) {
                *w -= learning_rate * g;
            }
            self.expert_biases[expert] -= learning_rate * db[expert];
        }
        self.loss(data)?;
        Ok(0)
    }
}
