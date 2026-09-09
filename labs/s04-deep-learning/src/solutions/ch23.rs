//! Full unrolling keeps one state per time, and reverse-mode sums every use of each shared weight.
//! LSTM backward carries TWO future derivatives: exposed hidden state and additive cell memory.
use crate::sequence::{self, Core, Lstm, LstmGradient, Rnn, RnnCache, RnnGradient};
pub(crate) const CORE: Core = Core {
    rnn_forward,
    rnn_gradient,
    lstm_gradient,
    rollout,
};
pub(crate) fn rnn_forward(model: &Rnn, inputs: &[f64]) -> RnnCache {
    let mut states = Vec::with_capacity(inputs.len() + 1);
    states.push(0.0);
    let mut predictions = Vec::with_capacity(inputs.len());
    for (t, &input) in inputs.iter().enumerate() {
        let state =
            (model.input_weight * input + model.recurrent_weight * states[t] + model.bias).tanh();
        states.push(state);
        predictions.push(model.output_weight * state + model.output_bias);
    }
    RnnCache {
        states,
        predictions,
    }
}
pub(crate) fn rnn_gradient(model: &Rnn, values: &[f64]) -> (f64, RnnGradient) {
    assert!(
        values.len() >= 2 && values.iter().all(|x| x.is_finite()),
        "sequence needs two finite observations"
    );
    let target_count = values.len() - 1;
    let inputs = &values[..target_count];
    let targets = &values[1..];
    let cache = model.forward(inputs);
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
        let prediction_gradient = 2.0 * (cache.predictions[t] - targets[t]) / target_count as f64;
        gradient.output_weight += prediction_gradient * cache.states[t + 1];
        gradient.output_bias += prediction_gradient;
        let preactivation_gradient = (prediction_gradient * model.output_weight
            + next_state_gradient)
            * (1.0 - cache.states[t + 1].powi(2));
        gradient.input_weight += preactivation_gradient * inputs[t];
        gradient.recurrent_weight += preactivation_gradient * cache.states[t];
        gradient.bias += preactivation_gradient;
        next_state_gradient = preactivation_gradient * model.recurrent_weight;
    }
    (loss, gradient)
}
pub(crate) fn lstm_gradient(model: &Lstm, values: &[f64]) -> (f64, LstmGradient) {
    assert!(
        values.len() >= 2 && values.iter().all(|x| x.is_finite()),
        "sequence needs two finite observations"
    );
    let target_count = values.len() - 1;
    let inputs = &values[..target_count];
    let targets = &values[1..];
    let cache = model.forward(inputs);
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
        let hidden_gradient = prediction_gradient * model.output_weight + next_hidden_gradient;
        let tanh_c = step.c.tanh();
        let output_gradient = hidden_gradient * tanh_c;
        let cell_gradient = hidden_gradient * step.o * (1.0 - tanh_c * tanh_c) + next_cell_gradient;
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
        next_hidden_gradient = input_preactivation_gradient * model.input.recurrent_weight
            + forget_preactivation_gradient * model.forget.recurrent_weight
            + output_preactivation_gradient * model.output.recurrent_weight
            + candidate_preactivation_gradient * model.candidate.recurrent_weight;
    }
    (loss, gradient)
}
pub(crate) fn rollout(model: &Lstm, initial: f64, horizon: usize) -> Vec<f64> {
    // ponytail: O(horizon²) replay, bounded to40 by CLI; carry cached state for long sequences.
    // Replaying the generated prefix reproduces one continuous hidden/cell trajectory.
    let mut inputs = vec![initial];
    let mut predictions = Vec::with_capacity(horizon);
    for _ in 0..horizon {
        let cache = model.forward(&inputs);
        let prediction = cache[inputs.len() - 1].prediction;
        predictions.push(prediction);
        inputs.push(prediction);
    }
    predictions
}
pub fn run(args: &[String]) -> Result<(), String> {
    sequence::run(CORE, args)
}
pub fn check() -> Result<(), String> {
    sequence::check(CORE)
}
