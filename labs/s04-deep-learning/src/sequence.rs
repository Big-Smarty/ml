//! Trainable scalar RNN and LSTM forecasters with chronological evaluation.
#[derive(Clone, Copy)]
pub(crate) struct Core {
    pub rnn_forward: fn(&Rnn, &[f64]) -> RnnCache,
    pub rnn_gradient: fn(&Rnn, &[f64]) -> (f64, RnnGradient),
    pub lstm_gradient: fn(&Lstm, &[f64]) -> (f64, LstmGradient),
    pub rollout: fn(&Lstm, f64, usize) -> Vec<f64>,
}
fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|t| (t as f64 * 0.17).sin() + 0.25 * (t as f64 * 0.043).cos())
        .collect()
}

#[derive(Clone, Copy)]
pub(crate) struct Rnn {
    core: Core,
    pub(crate) input_weight: f64,
    pub(crate) recurrent_weight: f64,
    pub(crate) bias: f64,
    pub(crate) output_weight: f64,
    pub(crate) output_bias: f64,
}
#[derive(Default)]
pub(crate) struct RnnGradient {
    pub(crate) input_weight: f64,
    pub(crate) recurrent_weight: f64,
    pub(crate) bias: f64,
    pub(crate) output_weight: f64,
    pub(crate) output_bias: f64,
}

pub(crate) struct RnnCache {
    pub(crate) states: Vec<f64>,
    pub(crate) predictions: Vec<f64>,
}

impl Rnn {
    fn new(core: Core) -> Self {
        Self {
            core,
            input_weight: 0.3,
            recurrent_weight: 0.2,
            bias: 0.0,
            output_weight: 0.4,
            output_bias: 0.0,
        }
    }
    pub(crate) fn forward(&self, inputs: &[f64]) -> RnnCache {
        (self.core.rnn_forward)(self, inputs)
    }
    fn loss_and_gradient(&self, values: &[f64]) -> (f64, RnnGradient) {
        (self.core.rnn_gradient)(self, values)
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
pub(crate) struct Gate {
    pub(crate) input_weight: f64,
    pub(crate) recurrent_weight: f64,
    pub(crate) bias: f64,
}
#[derive(Clone, Copy)]
pub(crate) struct Lstm {
    core: Core,
    pub(crate) input: Gate,
    pub(crate) forget: Gate,
    pub(crate) output: Gate,
    pub(crate) candidate: Gate,
    pub(crate) output_weight: f64,
    pub(crate) output_bias: f64,
}
#[derive(Default, Clone, Copy)]
pub(crate) struct GateGradient {
    pub(crate) input_weight: f64,
    pub(crate) recurrent_weight: f64,
    pub(crate) bias: f64,
}
#[derive(Default)]
pub(crate) struct LstmGradient {
    pub(crate) input: GateGradient,
    pub(crate) forget: GateGradient,
    pub(crate) output: GateGradient,
    pub(crate) candidate: GateGradient,
    pub(crate) output_weight: f64,
    pub(crate) output_bias: f64,
}
#[derive(Clone, Copy)]
pub(crate) struct LstmStep {
    pub(crate) input: f64,
    pub(crate) h_prev: f64,
    pub(crate) c_prev: f64,
    pub(crate) i: f64,
    pub(crate) f: f64,
    pub(crate) o: f64,
    pub(crate) g: f64,
    pub(crate) c: f64,
    pub(crate) h: f64,
    pub(crate) prediction: f64,
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
    fn new(core: Core) -> Self {
        Self {
            core,
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
    pub(crate) fn forward(&self, inputs: &[f64]) -> Vec<LstmStep> {
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
        (self.core.lstm_gradient)(self, values)
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

pub(crate) fn run(core: Core, args: &[String]) -> Result<(), String> {
    if let [flag, stage] = args {
        if flag == "--checkpoint" {
            return check_stage(core, stage);
        }
    }
    let horizon = match args {
        [] => 10,
        [flag, value] if flag == "--horizon" => value
            .parse::<usize>()
            .map_err(|_| "horizon must be an integer in1..40")?,
        _ => return Err("usage: 23 [--horizon 1..40]".into()),
    };
    if !(1..=40).contains(&horizon) {
        return Err("horizon must be in1..40".into());
    }
    let series = signal(121);
    let (train_values, validation_values) = split(&series, 80);
    println!(
        "signal inputs {:?}; training targets1..80, validation targets81..120; zero initial state",
        &series[..4]
    );
    println!("RNN epochs=700, learning_rate=0.03; LSTM epochs=900, learning_rate=0.04; selected deployment is baseline persistence or completed feedback");
    let baseline = persistence_mse(validation_values);
    let mut rnn = Rnn::new(core);
    let rnn_before = rnn.loss(validation_values);
    rnn.train(train_values, 700, 0.03);
    let rnn_after = rnn.loss(validation_values);
    let mut lstm = Lstm::new(core);
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
    let mut history = vec![validation_values[0]];
    // ponytail: O(horizon²) replay is bounded to40; cache states for long rollout.
    for _ in 0..horizon {
        let prediction = rnn.forward(&history).predictions[history.len() - 1];
        history.push(prediction);
    }
    let rollout_mse = history[1..]
        .iter()
        .zip(&validation_values[1..=horizon])
        .map(|(p, y)| (p - y).powi(2))
        .sum::<f64>()
        / horizon as f64;
    println!(
        "RNN free-running {horizon}-horizon mean squared error {rollout_mse:.5}; first predictions {:?}",
        &history[1..=horizon.min(3)]
    );
    let deployed = (core.rollout)(&lstm, validation_values[0], horizon);
    println!("horizon | target | RNN rollout squared error | selected LSTM deployment squared error | persistence squared error");
    for (t, (prediction, target)) in deployed
        .iter()
        .zip(&validation_values[1..=horizon])
        .enumerate()
    {
        println!(
            "{} | {:.5} | {:.5} | {:.5} | {:.5}",
            t + 1,
            target,
            (history[t + 1] - target).powi(2),
            (prediction - target).powi(2),
            (validation_values[0] - target).powi(2)
        );
    }
    Ok(())
}
pub(crate) fn check(core: Core) -> Result<(), String> {
    check_stage(core, "all")
}
fn check_stage(core: Core, stage: &str) -> Result<(), String> {
    if !["states", "bptt", "lstm", "rollout", "training", "all"].contains(&stage) {
        return Err(format!("unknown chapter23 checkpoint {stage}"));
    }

    let values = [0.2, -0.4, 0.7, 0.1, -0.3];
    let rnn = Rnn::new(core);
    let cache = rnn.forward(&values[..4]);
    let h1 = (0.3_f64 * 0.2).tanh();
    let h2 = (-0.3_f64 * 0.4 + 0.2 * h1).tanh();
    if cache.states.len() != 5 || cache.predictions.len() != 4 {
        return Err(
            "GOAL_NOT_MET: RNN cache needs N+1 states including zero and N predictions".into(),
        );
    }
    crate::close("state must carry across time", cache.states[2], h2)?;
    if stage == "states" {
        println!("PASS states checkpoint");
        return Ok(());
    }
    let analytic = rnn.loss_and_gradient(&values).1;
    let mut plus = rnn;
    let mut minus = rnn;
    plus.recurrent_weight += 1e-5;
    minus.recurrent_weight -= 1e-5;
    crate::close(
        "RNN recurrent BPTT",
        analytic.recurrent_weight,
        (plus.loss(&values) - minus.loss(&values)) / 2e-5,
    )?;
    plus = rnn;
    minus = rnn;
    plus.input_weight += 1e-5;
    minus.input_weight -= 1e-5;
    crate::close(
        "RNN input BPTT",
        analytic.input_weight,
        (plus.loss(&values) - minus.loss(&values)) / 2e-5,
    )?;
    if stage == "bptt" {
        println!("PASS bptt checkpoint");
        return Ok(());
    }
    let lstm = Lstm::new(core);
    let analytic = lstm.loss_and_gradient(&values).1;
    let mut plus = lstm;
    let mut minus = lstm;
    plus.candidate.input_weight += 1e-5;
    minus.candidate.input_weight -= 1e-5;
    crate::close(
        "LSTM candidate through later cells",
        analytic.candidate.input_weight,
        (plus.loss(&values) - minus.loss(&values)) / 2e-5,
    )?;
    plus = lstm;
    minus = lstm;
    plus.forget.bias += 1e-5;
    minus.forget.bias -= 1e-5;
    crate::close(
        "LSTM forget bias",
        analytic.forget.bias,
        (plus.loss(&values) - minus.loss(&values)) / 2e-5,
    )?;
    if stage == "lstm" {
        println!("PASS lstm checkpoint");
        return Ok(());
    }
    let seed = 0.63;
    let first = lstm.forward(&[seed])[0].prediction;
    let second = lstm.forward(&[seed, first])[1].prediction;
    let forecast = (core.rollout)(&lstm, seed, 2);
    if forecast.len() != 2 {
        return Err(
            "GOAL_NOT_MET: rollout must return exactly the requested number of predictions".into(),
        );
    }
    crate::close("rollout first prediction", forecast[0], first)?;
    crate::close(
        "rollout feeds generated output with carried state",
        forecast[1],
        second,
    )?;
    if stage == "rollout" {
        println!("PASS free-running LSTM feedback and state");
        return Ok(());
    }
    let series = signal(121);
    let (train, valid) = split(&series, 80);
    let mut rnn = Rnn::new(core);
    let mut lstm = Lstm::new(core);
    let before_r = rnn.loss(valid);
    let before_l = lstm.loss(valid);
    rnn.train(train, 700, 0.03);
    lstm.train(train, 900, 0.04);
    if rnn.loss(valid) >= before_r * 0.5 || lstm.loss(valid) >= before_l * 0.5 {
        return Err(
            "GOAL_NOT_MET: trained sequence models must improve their initial held-out MSE".into(),
        );
    }
    println!("PASS ordered states, RNN and LSTM BPTT, unfamiliar signed sequence; later MSE RNN {:.5}, LSTM {:.5}, persistence {:.5}",rnn.loss(valid),lstm.loss(valid),persistence_mse(valid));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rnn_recurrent_gradient_matches_central_difference() {
        let values = signal(9);
        let mut model = Rnn::new(crate::solutions::ch23::CORE);
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
        assert!(
            std::panic::catch_unwind(|| Rnn::new(crate::solutions::ch23::CORE).loss(&[1.0]))
                .is_err()
        );
        assert!(std::panic::catch_unwind(
            || Lstm::new(crate::solutions::ch23::CORE).loss(&[0.0, f64::NAN])
        )
        .is_err());
    }

    #[test]
    fn lstm_bptt_gradient_matches_central_difference() {
        let values = signal(9);
        let mut model = Lstm::new(crate::solutions::ch23::CORE);
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
        let mut rnn = Rnn::new(crate::solutions::ch23::CORE);
        let rb = rnn.loss(validation_values);
        rnn.train(train_values, 700, 0.03);
        let mut lstm = Lstm::new(crate::solutions::ch23::CORE);
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
