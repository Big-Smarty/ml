//! Trainable scalar RNN and LSTM forecasters with chronological evaluation.
fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|t| (t as f64 * 0.17).sin() + 0.25 * (t as f64 * 0.043).cos())
        .collect()
}

#[derive(Clone, Copy)]
struct Rnn {
    input_weight: f64,
    recurrent_weight: f64,
    bias: f64,
    output_weight: f64,
    output_bias: f64,
}
#[derive(Default)]
struct RnnGradient {
    input_weight: f64,
    recurrent_weight: f64,
    bias: f64,
    output_weight: f64,
    output_bias: f64,
}

struct RnnCache {
    states: Vec<f64>,
    predictions: Vec<f64>,
}

impl Rnn {
    fn new() -> Self {
        Self {
            input_weight: 0.3,
            recurrent_weight: 0.2,
            bias: 0.0,
            output_weight: 0.4,
            output_bias: 0.0,
        }
    }
    fn forward(&self, inputs: &[f64]) -> RnnCache {
        let mut states = Vec::with_capacity(inputs.len() + 1);
        states.push(0.0);
        let mut predictions = Vec::with_capacity(inputs.len());
        for (t, &input) in inputs.iter().enumerate() {
            let state =
                (self.input_weight * input + self.recurrent_weight * states[t] + self.bias).tanh();
            states.push(state);
            predictions.push(self.output_weight * state + self.output_bias);
        }
        RnnCache {
            states,
            predictions,
        }
    }
    fn loss_and_gradient(&self, values: &[f64]) -> (f64, RnnGradient) {
        assert!(
            values.len() >= 2 && values.iter().all(|x| x.is_finite()),
            "sequence needs two finite observations"
        );
        let target_count = values.len() - 1;
        let inputs = &values[..target_count];
        let targets = &values[1..];
        let cache = self.forward(inputs);
        let loss = cache
            .predictions
            .iter()
            .zip(targets)
            .map(|(prediction, target)| (prediction - target).powi(2))
            .sum::<f64>()
            / target_count as f64;
        let mut gradient = RnnGradient::default();
        let mut next_state_gradient = 0.0;
        for t in (0..target_count).rev() {
            let prediction_gradient =
                2.0 * (cache.predictions[t] - targets[t]) / target_count as f64;
            gradient.output_weight += prediction_gradient * cache.states[t + 1];
            gradient.output_bias += prediction_gradient;
            let preactivation_gradient = (prediction_gradient * self.output_weight
                + next_state_gradient)
                * (1.0 - cache.states[t + 1].powi(2));
            gradient.input_weight += preactivation_gradient * inputs[t];
            gradient.recurrent_weight += preactivation_gradient * cache.states[t];
            gradient.bias += preactivation_gradient;
            next_state_gradient = preactivation_gradient * self.recurrent_weight;
        }
        (loss, gradient)
    }
    fn train(&mut self, values: &[f64], epochs: usize, learning_rate: f64) {
        assert!(
            epochs > 0 && learning_rate.is_finite() && learning_rate > 0.0,
            "positive finite training settings required"
        );
        for _ in 0..epochs {
            let (_, gradient) = self.loss_and_gradient(values);
            self.input_weight -= learning_rate * gradient.input_weight;
            self.recurrent_weight -= learning_rate * gradient.recurrent_weight;
            self.bias -= learning_rate * gradient.bias;
            self.output_weight -= learning_rate * gradient.output_weight;
            self.output_bias -= learning_rate * gradient.output_bias;
        }
    }
    fn loss(&self, values: &[f64]) -> f64 {
        self.loss_and_gradient(values).0
    }
}

#[derive(Clone, Copy)]
struct Gate {
    input_weight: f64,
    recurrent_weight: f64,
    bias: f64,
}
#[derive(Clone, Copy)]
struct Lstm {
    input: Gate,
    forget: Gate,
    output: Gate,
    candidate: Gate,
    output_weight: f64,
    output_bias: f64,
}
#[derive(Default, Clone, Copy)]
struct GateGradient {
    input_weight: f64,
    recurrent_weight: f64,
    bias: f64,
}
#[derive(Default)]
struct LstmGradient {
    input: GateGradient,
    forget: GateGradient,
    output: GateGradient,
    candidate: GateGradient,
    output_weight: f64,
    output_bias: f64,
}
#[derive(Clone, Copy)]
struct LstmStep {
    input: f64,
    h_prev: f64,
    c_prev: f64,
    i: f64,
    f: f64,
    o: f64,
    g: f64,
    c: f64,
    h: f64,
    prediction: f64,
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

impl Lstm {
    fn new() -> Self {
        Self {
            input: Gate {
                input_weight: 0.4,
                recurrent_weight: 0.1,
                bias: 0.0,
            },
            forget: Gate {
                input_weight: -0.2,
                recurrent_weight: 0.1,
                bias: 1.0,
            },
            output: Gate {
                input_weight: 0.3,
                recurrent_weight: -0.1,
                bias: 0.0,
            },
            candidate: Gate {
                input_weight: 0.5,
                recurrent_weight: 0.2,
                bias: 0.0,
            },
            output_weight: 0.6,
            output_bias: 0.0,
        }
    }
    fn gate(gate: Gate, input: f64, previous_state: f64) -> f64 {
        sigmoid(gate.input_weight * input + gate.recurrent_weight * previous_state + gate.bias)
    }
    fn forward(&self, inputs: &[f64]) -> Vec<LstmStep> {
        let (mut hidden, mut cell) = (0.0, 0.0);
        let mut cache = Vec::with_capacity(inputs.len());
        for &input in inputs {
            let h_prev = hidden;
            let c_prev = cell;
            let i = Self::gate(self.input, input, h_prev);
            let f = Self::gate(self.forget, input, h_prev);
            let o = Self::gate(self.output, input, h_prev);
            let g = (self.candidate.input_weight * input
                + self.candidate.recurrent_weight * h_prev
                + self.candidate.bias)
                .tanh();
            cell = f * c_prev + i * g;
            hidden = o * cell.tanh();
            cache.push(LstmStep {
                input,
                h_prev,
                c_prev,
                i,
                f,
                o,
                g,
                c: cell,
                h: hidden,
                prediction: self.output_weight * hidden + self.output_bias,
            });
        }
        cache
    }
    fn loss_and_gradient(&self, values: &[f64]) -> (f64, LstmGradient) {
        assert!(
            values.len() >= 2 && values.iter().all(|x| x.is_finite()),
            "sequence needs two finite observations"
        );
        let target_count = values.len() - 1;
        let inputs = &values[..target_count];
        let targets = &values[1..];
        let cache = self.forward(inputs);
        let loss = cache
            .iter()
            .zip(targets)
            .map(|(step, target)| (step.prediction - target).powi(2))
            .sum::<f64>()
            / target_count as f64;
        let mut gradient = LstmGradient::default();
        let (mut next_hidden_gradient, mut next_cell_gradient) = (0.0, 0.0);
        for t in (0..target_count).rev() {
            let step = cache[t];
            let prediction_gradient = 2.0 * (step.prediction - targets[t]) / target_count as f64;
            gradient.output_weight += prediction_gradient * step.h;
            gradient.output_bias += prediction_gradient;
            let hidden_gradient = prediction_gradient * self.output_weight + next_hidden_gradient;
            let tanh_c = step.c.tanh();
            let output_gradient = hidden_gradient * tanh_c;
            let cell_gradient =
                hidden_gradient * step.o * (1.0 - tanh_c * tanh_c) + next_cell_gradient;
            let forget_gradient = cell_gradient * step.c_prev;
            let input_gradient = cell_gradient * step.g;
            let candidate_gradient = cell_gradient * step.i;
            next_cell_gradient = cell_gradient * step.f;
            let input_preactivation_gradient = input_gradient * step.i * (1.0 - step.i);
            let forget_preactivation_gradient = forget_gradient * step.f * (1.0 - step.f);
            let output_preactivation_gradient = output_gradient * step.o * (1.0 - step.o);
            let candidate_preactivation_gradient = candidate_gradient * (1.0 - step.g * step.g);
            for (gate_gradient, preactivation_gradient) in [
                (&mut gradient.input, input_preactivation_gradient),
                (&mut gradient.forget, forget_preactivation_gradient),
                (&mut gradient.output, output_preactivation_gradient),
                (&mut gradient.candidate, candidate_preactivation_gradient),
            ] {
                gate_gradient.input_weight += preactivation_gradient * step.input;
                gate_gradient.recurrent_weight += preactivation_gradient * step.h_prev;
                gate_gradient.bias += preactivation_gradient;
            }
            next_hidden_gradient = input_preactivation_gradient * self.input.recurrent_weight
                + forget_preactivation_gradient * self.forget.recurrent_weight
                + output_preactivation_gradient * self.output.recurrent_weight
                + candidate_preactivation_gradient * self.candidate.recurrent_weight;
        }
        (loss, gradient)
    }
    fn apply_gate(gate: &mut Gate, gradient: GateGradient, learning_rate: f64) {
        gate.input_weight -= learning_rate * gradient.input_weight;
        gate.recurrent_weight -= learning_rate * gradient.recurrent_weight;
        gate.bias -= learning_rate * gradient.bias;
    }
    fn train(&mut self, values: &[f64], epochs: usize, learning_rate: f64) {
        assert!(
            epochs > 0 && learning_rate.is_finite() && learning_rate > 0.0,
            "positive finite training settings required"
        );
        for _ in 0..epochs {
            let (_, gradient) = self.loss_and_gradient(values);
            Self::apply_gate(&mut self.input, gradient.input, learning_rate);
            Self::apply_gate(&mut self.forget, gradient.forget, learning_rate);
            Self::apply_gate(&mut self.output, gradient.output, learning_rate);
            Self::apply_gate(&mut self.candidate, gradient.candidate, learning_rate);
            self.output_weight -= learning_rate * gradient.output_weight;
            self.output_bias -= learning_rate * gradient.output_bias;
        }
    }
    fn loss(&self, values: &[f64]) -> f64 {
        self.loss_and_gradient(values).0
    }
}

fn split(series: &[f64], cut: usize) -> (&[f64], &[f64]) {
    // Validation starts with the last training value so its first target is truly future.
    (&series[..=cut], &series[cut..])
}

fn persistence_mse(values: &[f64]) -> f64 {
    values
        .windows(2)
        .map(|pair| (pair[0] - pair[1]).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64
}

fn main() {
    let series = signal(121);
    let (train_values, validation_values) = split(&series, 80);
    let baseline = persistence_mse(validation_values);
    let mut rnn = Rnn::new();
    let rnn_before = rnn.loss(validation_values);
    rnn.train(train_values, 700, 0.03);
    let rnn_after = rnn.loss(validation_values);
    let mut lstm = Lstm::new();
    let lstm_before = lstm.loss(validation_values);
    lstm.train(train_values, 900, 0.04);
    let lstm_after = lstm.loss(validation_values);
    println!(
        "chronological split: {} training targets, {} later validation targets",
        train_values.len() - 1,
        validation_values.len() - 1
    );
    println!("persistence validation one-step MSE {baseline:.5}");
    println!("RNN validation one-step MSE {rnn_before:.5} -> {rnn_after:.5}");
    println!("LSTM validation one-step MSE {lstm_before:.5} -> {lstm_after:.5}");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rnn_recurrent_gradient_matches_central_difference() {
        let values = signal(9);
        let mut model = Rnn::new();
        let analytic = model.loss_and_gradient(&values).1.recurrent_weight;
        let h = 1e-5;
        model.recurrent_weight += h;
        let plus = model.loss(&values);
        model.recurrent_weight -= 2.0 * h;
        let minus = model.loss(&values);
        let numeric = (plus - minus) / (2.0 * h);
        assert!((analytic - numeric).abs() < 1e-6 + 1e-4 * numeric.abs());
    }
    #[test]
    fn short_or_nonfinite_sequences_are_rejected() {
        assert!(std::panic::catch_unwind(|| Rnn::new().loss(&[1.0])).is_err());
        assert!(std::panic::catch_unwind(|| Lstm::new().loss(&[0.0, f64::NAN])).is_err());
    }

    #[test]
    fn lstm_bptt_gradient_matches_central_difference() {
        let values = signal(9);
        let mut model = Lstm::new();
        let analytic = model.loss_and_gradient(&values).1.candidate.input_weight;
        let h = 1e-5;
        model.candidate.input_weight += h;
        let plus = model.loss(&values);
        model.candidate.input_weight -= 2.0 * h;
        let minus = model.loss(&values);
        let numeric = (plus - minus) / (2.0 * h);
        assert!(
            (analytic - numeric).abs() < 1e-6 + 1e-4 * numeric.abs(),
            "{analytic} {numeric}"
        );
    }
    #[test]
    fn both_recurrent_models_learn_prefix_and_improve_future() {
        let series = signal(121);
        let (train_values, validation_values) = split(&series, 80);
        let mut rnn = Rnn::new();
        let rb = rnn.loss(validation_values);
        rnn.train(train_values, 700, 0.03);
        let mut lstm = Lstm::new();
        let lb = lstm.loss(validation_values);
        lstm.train(train_values, 900, 0.04);
        assert!(rnn.loss(validation_values) < rb * 0.5);
        assert!(lstm.loss(validation_values) < lb * 0.5);
    }
    #[test]
    fn temporal_split_has_no_future_training_targets() {
        let series = signal(121);
        let (train_values, validation_values) = split(&series, 80);
        assert_eq!(train_values.last(), validation_values.first());
        assert_eq!(
            (train_values.len() - 1, validation_values.len() - 1),
            (80, 40)
        );
    }
}
