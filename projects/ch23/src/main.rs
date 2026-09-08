//! Trainable scalar RNN and LSTM forecasters with chronological evaluation.
fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|t| (t as f64 * 0.17).sin() + 0.25 * (t as f64 * 0.043).cos())
        .collect()
}

#[derive(Clone, Copy)]
struct Rnn {
    wx: f64,
    wh: f64,
    b: f64,
    wy: f64,
    by: f64,
}
#[derive(Default)]
struct RnnGrad {
    wx: f64,
    wh: f64,
    b: f64,
    wy: f64,
    by: f64,
}

impl Rnn {
    fn new() -> Self {
        Self {
            wx: 0.3,
            wh: 0.2,
            b: 0.0,
            wy: 0.4,
            by: 0.0,
        }
    }
    fn loss_grad(&self, values: &[f64]) -> (f64, RnnGrad) {
        assert!(
            values.len() >= 2 && values.iter().all(|x| x.is_finite()),
            "sequence needs two finite observations"
        );
        let n = values.len() - 1;
        let mut states = Vec::with_capacity(n + 1);
        states.push(0.0);
        let mut predictions = Vec::with_capacity(n);
        for t in 0..n {
            let h = (self.wx * values[t] + self.wh * states[t] + self.b).tanh();
            states.push(h);
            predictions.push(self.wy * h + self.by);
        }
        let loss = predictions
            .iter()
            .enumerate()
            .map(|(t, p)| (p - values[t + 1]).powi(2))
            .sum::<f64>()
            / n as f64;
        let mut g = RnnGrad::default();
        let mut dh_next = 0.0;
        for t in (0..n).rev() {
            let dp = 2.0 * (predictions[t] - values[t + 1]) / n as f64;
            g.wy += dp * states[t + 1];
            g.by += dp;
            let da = (dp * self.wy + dh_next) * (1.0 - states[t + 1].powi(2));
            g.wx += da * values[t];
            g.wh += da * states[t];
            g.b += da;
            dh_next = da * self.wh;
        }
        (loss, g)
    }
    fn train(&mut self, values: &[f64], epochs: usize, rate: f64) {
        assert!(
            epochs > 0 && rate.is_finite() && rate > 0.0,
            "positive finite training settings required"
        );
        for _ in 0..epochs {
            let (_, g) = self.loss_grad(values);
            self.wx -= rate * g.wx;
            self.wh -= rate * g.wh;
            self.b -= rate * g.b;
            self.wy -= rate * g.wy;
            self.by -= rate * g.by;
        }
    }
    fn mse(&self, values: &[f64]) -> f64 {
        self.loss_grad(values).0
    }
}

#[derive(Clone, Copy)]
struct Gate {
    wx: f64,
    wh: f64,
    b: f64,
}
#[derive(Clone, Copy)]
struct Lstm {
    input: Gate,
    forget: Gate,
    output: Gate,
    candidate: Gate,
    wy: f64,
    by: f64,
}
#[derive(Default, Clone, Copy)]
struct GateGrad {
    wx: f64,
    wh: f64,
    b: f64,
}
#[derive(Default)]
struct LstmGrad {
    input: GateGrad,
    forget: GateGrad,
    output: GateGrad,
    candidate: GateGrad,
    wy: f64,
    by: f64,
}
#[derive(Clone, Copy)]
struct Step {
    x: f64,
    h_prev: f64,
    c_prev: f64,
    i: f64,
    f: f64,
    o: f64,
    g: f64,
    c: f64,
    h: f64,
    pred: f64,
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
                wx: 0.4,
                wh: 0.1,
                b: 0.0,
            },
            forget: Gate {
                wx: -0.2,
                wh: 0.1,
                b: 1.0,
            },
            output: Gate {
                wx: 0.3,
                wh: -0.1,
                b: 0.0,
            },
            candidate: Gate {
                wx: 0.5,
                wh: 0.2,
                b: 0.0,
            },
            wy: 0.6,
            by: 0.0,
        }
    }
    fn gate(g: Gate, x: f64, h: f64) -> f64 {
        sigmoid(g.wx * x + g.wh * h + g.b)
    }
    fn loss_grad(&self, values: &[f64]) -> (f64, LstmGrad) {
        assert!(
            values.len() >= 2 && values.iter().all(|x| x.is_finite()),
            "sequence needs two finite observations"
        );
        let n = values.len() - 1;
        let (mut h, mut c) = (0.0, 0.0);
        let mut steps = Vec::with_capacity(n);
        for &x in values.iter().take(n) {
            let h_prev = h;
            let c_prev = c;
            let i = Self::gate(self.input, x, h_prev);
            let f = Self::gate(self.forget, x, h_prev);
            let o = Self::gate(self.output, x, h_prev);
            let g = (self.candidate.wx * x + self.candidate.wh * h_prev + self.candidate.b).tanh();
            c = f * c_prev + i * g;
            h = o * c.tanh();
            steps.push(Step {
                x,
                h_prev,
                c_prev,
                i,
                f,
                o,
                g,
                c,
                h,
                pred: self.wy * h + self.by,
            });
        }
        let loss = steps
            .iter()
            .enumerate()
            .map(|(t, s)| (s.pred - values[t + 1]).powi(2))
            .sum::<f64>()
            / n as f64;
        let mut grad = LstmGrad::default();
        let (mut dh_next, mut dc_next) = (0.0, 0.0);
        for t in (0..n).rev() {
            let s = steps[t];
            let dp = 2.0 * (s.pred - values[t + 1]) / n as f64;
            grad.wy += dp * s.h;
            grad.by += dp;
            let dh = dp * self.wy + dh_next;
            let tanh_c = s.c.tanh();
            let d_o = dh * tanh_c;
            let dc = dh * s.o * (1.0 - tanh_c * tanh_c) + dc_next;
            let d_f = dc * s.c_prev;
            let d_i = dc * s.g;
            let d_g = dc * s.i;
            dc_next = dc * s.f;
            let da_i = d_i * s.i * (1.0 - s.i);
            let da_f = d_f * s.f * (1.0 - s.f);
            let da_o = d_o * s.o * (1.0 - s.o);
            let da_g = d_g * (1.0 - s.g * s.g);
            for (gg, da) in [
                (&mut grad.input, da_i),
                (&mut grad.forget, da_f),
                (&mut grad.output, da_o),
                (&mut grad.candidate, da_g),
            ] {
                gg.wx += da * s.x;
                gg.wh += da * s.h_prev;
                gg.b += da;
            }
            dh_next = da_i * self.input.wh
                + da_f * self.forget.wh
                + da_o * self.output.wh
                + da_g * self.candidate.wh;
        }
        (loss, grad)
    }
    fn apply_gate(g: &mut Gate, dg: GateGrad, rate: f64) {
        g.wx -= rate * dg.wx;
        g.wh -= rate * dg.wh;
        g.b -= rate * dg.b;
    }
    fn train(&mut self, values: &[f64], epochs: usize, rate: f64) {
        assert!(
            epochs > 0 && rate.is_finite() && rate > 0.0,
            "positive finite training settings required"
        );
        for _ in 0..epochs {
            let (_, g) = self.loss_grad(values);
            Self::apply_gate(&mut self.input, g.input, rate);
            Self::apply_gate(&mut self.forget, g.forget, rate);
            Self::apply_gate(&mut self.output, g.output, rate);
            Self::apply_gate(&mut self.candidate, g.candidate, rate);
            self.wy -= rate * g.wy;
            self.by -= rate * g.by;
        }
    }
    fn mse(&self, values: &[f64]) -> f64 {
        self.loss_grad(values).0
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
    let (train, valid) = split(&series, 80);
    let baseline = persistence_mse(valid);
    let mut rnn = Rnn::new();
    let rnn_before = rnn.mse(valid);
    rnn.train(train, 700, 0.03);
    let rnn_after = rnn.mse(valid);
    let mut lstm = Lstm::new();
    let lstm_before = lstm.mse(valid);
    lstm.train(train, 900, 0.04);
    let lstm_after = lstm.mse(valid);
    println!(
        "chronological split: {} training targets, {} later validation targets",
        train.len() - 1,
        valid.len() - 1
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
        let analytic = model.loss_grad(&values).1.wh;
        let h = 1e-5;
        model.wh += h;
        let plus = model.mse(&values);
        model.wh -= 2.0 * h;
        let minus = model.mse(&values);
        let numeric = (plus - minus) / (2.0 * h);
        assert!((analytic - numeric).abs() < 1e-6 + 1e-4 * numeric.abs());
    }
    #[test]
    fn short_or_nonfinite_sequences_are_rejected() {
        assert!(std::panic::catch_unwind(|| Rnn::new().mse(&[1.0])).is_err());
        assert!(std::panic::catch_unwind(|| Lstm::new().mse(&[0.0, f64::NAN])).is_err());
    }

    #[test]
    fn lstm_bptt_gradient_matches_central_difference() {
        let values = signal(9);
        let mut model = Lstm::new();
        let analytic = model.loss_grad(&values).1.candidate.wx;
        let h = 1e-5;
        model.candidate.wx += h;
        let plus = model.mse(&values);
        model.candidate.wx -= 2.0 * h;
        let minus = model.mse(&values);
        let numeric = (plus - minus) / (2.0 * h);
        assert!(
            (analytic - numeric).abs() < 1e-6 + 1e-4 * numeric.abs(),
            "{analytic} {numeric}"
        );
    }
    #[test]
    fn both_recurrent_models_learn_prefix_and_improve_future() {
        let series = signal(121);
        let (train, valid) = split(&series, 80);
        let mut rnn = Rnn::new();
        let rb = rnn.mse(valid);
        rnn.train(train, 700, 0.03);
        let mut lstm = Lstm::new();
        let lb = lstm.mse(valid);
        lstm.train(train, 900, 0.04);
        assert!(rnn.mse(valid) < rb * 0.5);
        assert!(lstm.mse(valid) < lb * 0.5);
    }
    #[test]
    fn temporal_split_has_no_future_training_targets() {
        let series = signal(121);
        let (train, valid) = split(&series, 80);
        assert_eq!(train.last(), valid.first());
        assert_eq!((train.len() - 1, valid.len() - 1), (80, 40));
    }
}
