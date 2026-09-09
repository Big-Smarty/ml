//! Working baseline: a nonlinear one-lag RNN predictor and length-one truncated LSTM training.
//! Build the full RNN unroll/reverse pass, then extend LSTM credit across both state paths.
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
    for &input in inputs {
        let state = (model.input_weight * input + model.bias).tanh();
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

    for t in (0..target_count).rev() {
        let prediction_gradient = 2.0 * (cache.predictions[t] - targets[t]) / target_count as f64;
        gradient.output_weight += prediction_gradient * cache.states[t + 1];
        gradient.output_bias += prediction_gradient;
        let preactivation_gradient =
            (prediction_gradient * model.output_weight) * (1.0 - cache.states[t + 1].powi(2));
        gradient.input_weight += preactivation_gradient * inputs[t];
        gradient.bias += preactivation_gradient;
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
    // Baseline detaches state after each time step (truncation length one).
    let (next_hidden_gradient, next_cell_gradient) = (0.0, 0.0);
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
        // Full BPTT must carry the future cell contribution here.
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
        // Full BPTT must also carry the sum through all four recurrent gate weights.
    }
    (loss, gradient)
}
pub(crate) fn rollout(_model: &Lstm, initial: f64, horizon: usize) -> Vec<f64> {
    // Working deployment baseline: persistence predicts the last observed value at every horizon.
    vec![initial; horizon]
}
pub fn run(args: &[String]) -> Result<(), String> {
    sequence::run(CORE, args)
}
pub fn check() -> Result<(), String> {
    sequence::check(CORE)
}
