// A trainable top-1 mixture-of-experts layer with capacity and load balancing.

const INPUTS: usize = 2;
const EXPERTS: usize = 3;

fn softmax(logits: [f64; EXPERTS]) -> [f64; EXPERTS] {
    let maximum = logits.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities = logits.map(|logit| (logit - maximum).exp());
    let sum: f64 = probabilities.iter().sum();
    probabilities.iter_mut().for_each(|value| *value /= sum);
    probabilities
}

fn selected_gate(probability: f64, expert_output: f64) -> f64 {
    probability * expert_output
}

fn half_squared_error(prediction: f64, target: f64) -> f64 {
    0.5 * (prediction - target).powi(2)
}

#[derive(Clone, Debug)]
struct Model {
    router_weights: [[f64; INPUTS]; EXPERTS],
    expert_weights: [[f64; INPUTS]; EXPERTS],
    expert_biases: [f64; EXPERTS],
}

impl Model {
    fn new() -> Self {
        Self {
            router_weights: [[0.0, 0.0], [0.01, -0.02], [-0.02, 0.01]],
            expert_weights: [[0.1, -0.1], [-0.1, 0.05], [0.05, 0.1]],
            expert_biases: [0.0; EXPERTS],
        }
    }

    fn expert(&self, index: usize, input: [f64; INPUTS]) -> f64 {
        self.expert_weights[index][0] * input[0]
            + self.expert_weights[index][1] * input[1]
            + self.expert_biases[index]
    }

    fn predict(&self, input: [f64; INPUTS]) -> f64 {
        let (selected, probabilities) = self.route(input);
        selected_gate(probabilities[selected], self.expert(selected, input))
    }

    /// Uncapped task-only mean half-squared error over all examples.
    fn loss(&self, data: &[([f64; INPUTS], f64)]) -> Result<f64, &'static str> {
        let model_finite = self
            .router_weights
            .iter()
            .chain(self.expert_weights.iter())
            .flatten()
            .chain(self.expert_biases.iter())
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
            .map(|&(input, target)| half_squared_error(self.predict(input), target))
            .sum::<f64>()
            / data.len() as f64;
        if !loss.is_finite() {
            return Err("loss became nonfinite");
        }
        Ok(loss)
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

fn train_fixture(epochs: usize) -> Result<(Model, f64, f64, usize), &'static str> {
    let data = fixture();
    let mut model = Model::new();
    let before = model.loss(&data)?;
    let mut dropped = 0;
    for epoch in 0..epochs {
        dropped = model.train_epoch(&data, 0.08, 1.25, 0.02, epoch % data.len())?;
    }
    let after = model.loss(&data)?;
    Ok((model, before, after, dropped))
}

fn demo() -> Result<(), Box<dyn std::error::Error>> {
    let (model, before, after, dropped) = train_fixture(1_500)?;
    let mut counts = [0; EXPERTS];
    fixture()
        .iter()
        .for_each(|&(input, _)| counts[model.route(input).0] += 1);
    println!("MoE uncapped task half-MSE: {before:.5} -> {after:.5}");
    println!("route counts={counts:?}, final-epoch overflow={dropped}");
    println!(
        "router={:?}\nexperts={:?}",
        model.router_weights, model.expert_weights
    );
    println!("This tiny CPU layer verifies routing gradients and capacity, not large-model quality or speed.");
    Ok(())
}
