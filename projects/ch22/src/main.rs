//! Two real representation-learning experiments: an autoencoder and InfoNCE training.
const INPUTS: usize = 4;
const LATENT: usize = 2;

const CLEAN: [[f64; INPUTS]; 6] = [
    [1.0, 0.9, 0.1, 0.0],
    [0.9, 1.0, 0.0, 0.1],
    [0.1, 0.0, 1.0, 0.9],
    [0.0, 0.1, 0.9, 1.0],
    [0.8, 0.7, 0.3, 0.2],
    [0.2, 0.3, 0.7, 0.8],
];
const VIEW_A: [[f64; INPUTS]; 6] = [
    [1.0, 0.8, 0.1, 0.0],
    [0.8, 1.0, 0.0, 0.2],
    [0.2, 0.0, 1.0, 0.8],
    [0.0, 0.2, 0.8, 1.0],
    [0.9, 0.6, 0.3, 0.1],
    [0.1, 0.3, 0.6, 0.9],
];
const VIEW_B: [[f64; INPUTS]; 6] = [
    [0.9, 1.0, 0.0, 0.1],
    [1.0, 0.9, 0.1, 0.0],
    [0.0, 0.1, 0.9, 1.0],
    [0.1, 0.0, 1.0, 0.9],
    [0.7, 0.8, 0.2, 0.3],
    [0.3, 0.2, 0.8, 0.7],
];

#[derive(Clone)]
struct Autoencoder {
    encoder: [[f64; INPUTS]; LATENT],
    encoder_bias: [f64; LATENT],
    decoder: [[f64; LATENT]; INPUTS],
    decoder_bias: [f64; INPUTS],
}

impl Autoencoder {
    fn new() -> Self {
        Self {
            encoder: [[0.21, -0.13, 0.08, 0.17], [-0.07, 0.18, 0.23, -0.11]],
            encoder_bias: [0.0; LATENT],
            decoder: [[0.14, -0.09], [-0.12, 0.20], [0.08, 0.17], [0.19, -0.10]],
            decoder_bias: [0.0; INPUTS],
        }
    }
    fn encode(&self, x: &[f64; INPUTS]) -> [f64; LATENT] {
        let mut z = self.encoder_bias;
        for (h, value) in z.iter_mut().enumerate() {
            for (j, &input) in x.iter().enumerate() {
                *value += self.encoder[h][j] * input;
            }
            *value = value.tanh();
        }
        z
    }
    fn reconstruct(&self, x: &[f64; INPUTS]) -> ([f64; LATENT], [f64; INPUTS]) {
        let z = self.encode(x);
        let mut out = self.decoder_bias;
        for (j, value) in out.iter_mut().enumerate() {
            for (h, &latent) in z.iter().enumerate() {
                *value += self.decoder[j][h] * latent;
            }
        }
        (z, out)
    }
    fn loss(&self, rows: &[[f64; INPUTS]]) -> f64 {
        rows.iter()
            .map(|x| {
                let (_, out) = self.reconstruct(x);
                out.iter().zip(x).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / INPUTS as f64
            })
            .sum::<f64>()
            / rows.len() as f64
    }
    fn encoder_gradient(&self, x: &[f64; INPUTS]) -> ([[f64; INPUTS]; LATENT], [f64; LATENT]) {
        let (z, out) = self.reconstruct(x);
        let d_out: [f64; INPUTS] = std::array::from_fn(|j| 2.0 * (out[j] - x[j]) / INPUTS as f64);
        let mut gradient = [[0.0; INPUTS]; LATENT];
        let mut bias_gradient = [0.0; LATENT];
        for h in 0..LATENT {
            let d_z: f64 = (0..INPUTS).map(|j| self.decoder[j][h] * d_out[j]).sum();
            let d_pre = d_z * (1.0 - z[h] * z[h]);
            bias_gradient[h] = d_pre;
            for j in 0..INPUTS {
                gradient[h][j] = d_pre * x[j];
            }
        }
        (gradient, bias_gradient)
    }
    fn train(
        &mut self,
        rows: &[[f64; INPUTS]],
        epochs: usize,
        rate: f64,
    ) -> Result<(), &'static str> {
        if rows.is_empty()
            || rows.iter().flatten().any(|x| !x.is_finite())
            || epochs == 0
            || rate <= 0.0
            || !rate.is_finite()
        {
            return Err("nonempty data, positive epochs and rate required");
        }
        for _ in 0..epochs {
            for x in rows {
                let (z, out) = self.reconstruct(x);
                if out.iter().chain(&z).any(|x| !x.is_finite()) {
                    return Err("autoencoder overflow; reduce the rate");
                }
                let (encoder_gradient, encoder_bias_gradient) = self.encoder_gradient(x);
                let mut d_out = [0.0; INPUTS];
                for j in 0..INPUTS {
                    d_out[j] = 2.0 * (out[j] - x[j]) / INPUTS as f64;
                }
                for (j, &output_gradient) in d_out.iter().enumerate() {
                    for (h, &latent) in z.iter().enumerate() {
                        self.decoder[j][h] -= rate * output_gradient * latent;
                    }
                    self.decoder_bias[j] -= rate * output_gradient;
                }
                for (h, row) in self.encoder.iter_mut().enumerate() {
                    for (j, weight) in row.iter_mut().enumerate() {
                        *weight -= rate * encoder_gradient[h][j];
                    }
                    self.encoder_bias[h] -= rate * encoder_bias_gradient[h];
                }
                if self
                    .encoder
                    .iter()
                    .flatten()
                    .chain(self.decoder.iter().flatten())
                    .chain(&self.encoder_bias)
                    .chain(&self.decoder_bias)
                    .any(|x| !x.is_finite())
                {
                    return Err("autoencoder update overflow; reduce the rate");
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Contrastive {
    w: [[f64; INPUTS]; LATENT],
}

impl Contrastive {
    fn new() -> Self {
        Self {
            w: [[0.31, -0.22, 0.17, 0.09], [-0.11, 0.27, 0.08, -0.24]],
        }
    }
    fn raw_and_embed(&self, x: &[f64; INPUTS]) -> ([f64; LATENT], f64, [f64; LATENT]) {
        let mut raw = [0.0; LATENT];
        for (h, value) in raw.iter_mut().enumerate() {
            for (j, &input) in x.iter().enumerate() {
                *value += self.w[h][j] * input;
            }
        }
        let norm = (raw.iter().map(|v| v * v).sum::<f64>() + 1e-8).sqrt();
        (raw, norm, raw.map(|v| v / norm))
    }
    fn embed(&self, x: &[f64; INPUTS]) -> [f64; LATENT] {
        self.raw_and_embed(x).2
    }
    fn dot(a: [f64; LATENT], b: [f64; LATENT]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
    fn loss(&self, a: &[[f64; INPUTS]], b: &[[f64; INPUTS]], temperature: f64) -> f64 {
        let ea: Vec<_> = a.iter().map(|x| self.embed(x)).collect();
        let eb: Vec<_> = b.iter().map(|x| self.embed(x)).collect();
        let mut total = 0.0;
        for i in 0..a.len() {
            let logits: Vec<_> = eb
                .iter()
                .map(|&candidate| Self::dot(ea[i], candidate) / temperature)
                .collect();
            let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            total += (m - logits[i]) + logits.iter().map(|&z| (z - m).exp()).sum::<f64>().ln();
        }
        total / a.len() as f64
    }
    fn loss_grad(
        &self,
        a: &[[f64; INPUTS]],
        b: &[[f64; INPUTS]],
        temperature: f64,
    ) -> (f64, [[f64; INPUTS]; LATENT]) {
        let ca: Vec<_> = a.iter().map(|x| self.raw_and_embed(x)).collect();
        let cb: Vec<_> = b.iter().map(|x| self.raw_and_embed(x)).collect();
        let mut d_ea = vec![[0.0; LATENT]; a.len()];
        let mut d_eb = vec![[0.0; LATENT]; b.len()];
        let mut loss = 0.0;
        for i in 0..a.len() {
            let logits: Vec<_> = cb
                .iter()
                .map(|candidate| Self::dot(ca[i].2, candidate.2) / temperature)
                .collect();
            let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = logits.iter().map(|&z| (z - m).exp()).sum();
            loss += (m - logits[i]) + sum.ln();
            for j in 0..b.len() {
                let mut dlogit = (logits[j] - m).exp() / sum;
                dlogit -= f64::from(i == j);
                dlogit /= a.len() as f64;
                for h in 0..LATENT {
                    d_ea[i][h] += dlogit * cb[j].2[h] / temperature;
                    d_eb[j][h] += dlogit * ca[i].2[h] / temperature;
                }
            }
        }
        let mut gradient = [[0.0; INPUTS]; LATENT];
        for (views, cache, d_embed) in [(a, &ca, &d_ea), (b, &cb, &d_eb)] {
            for n in 0..views.len() {
                let dot: f64 = d_embed[n].iter().zip(cache[n].0).map(|(d, r)| d * r).sum();
                for h in 0..LATENT {
                    let d_raw =
                        d_embed[n][h] / cache[n].1 - cache[n].0[h] * dot / cache[n].1.powi(3);
                    for j in 0..INPUTS {
                        gradient[h][j] += d_raw * views[n][j];
                    }
                }
            }
        }
        (loss / a.len() as f64, gradient)
    }
    fn train(
        &mut self,
        a: &[[f64; INPUTS]],
        b: &[[f64; INPUTS]],
        epochs: usize,
        rate: f64,
        temperature: f64,
    ) -> Result<(), &'static str> {
        if a.iter().chain(b).flatten().any(|x| !x.is_finite())
            || a.is_empty()
            || a.len() != b.len()
            || epochs == 0
            || !rate.is_finite()
            || rate <= 0.0
            || !temperature.is_finite()
            || temperature <= 0.0
        {
            return Err("paired nonempty views and positive settings required");
        }
        for _ in 0..epochs {
            let (loss, grad) = self.loss_grad(a, b, temperature);
            if !loss.is_finite() || grad.iter().flatten().any(|x| !x.is_finite()) {
                return Err(
                    "contrastive arithmetic overflow; increase temperature or reduce the rate",
                );
            }
            for (r, row) in self.w.iter_mut().enumerate() {
                for (c, weight) in row.iter_mut().enumerate() {
                    *weight -= rate * grad[r][c];
                    if !weight.is_finite() {
                        return Err("contrastive update overflow; reduce the rate");
                    }
                }
            }
        }
        Ok(())
    }
    fn retrieval_accuracy(&self, a: &[[f64; INPUTS]], b: &[[f64; INPUTS]]) -> f64 {
        let eb: Vec<_> = b.iter().map(|x| self.embed(x)).collect();
        let correct = a
            .iter()
            .enumerate()
            .filter(|(i, x)| {
                let e = self.embed(x);
                let best = (0..b.len())
                    .max_by(|&p, &q| Self::dot(e, eb[p]).total_cmp(&Self::dot(e, eb[q])))
                    .unwrap();
                best == *i
            })
            .count();
        correct as f64 / a.len() as f64
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ae = Autoencoder::new();
    let ae_before = ae.loss(&CLEAN);
    ae.train(&CLEAN, 500, 0.08)?;
    let ae_after = ae.loss(&CLEAN);
    let code = ae.encode(&CLEAN[0]);
    println!(
        "autoencoder reconstruction MSE {ae_before:.5} -> {ae_after:.5}; first code [{:.3}, {:.3}]",
        code[0], code[1]
    );

    let mut contrastive = Contrastive::new();
    let nce_before = contrastive.loss(&VIEW_A, &VIEW_B, 0.2);
    let retrieval_before = contrastive.retrieval_accuracy(&VIEW_A, &VIEW_B);
    contrastive.train(&VIEW_A, &VIEW_B, 120, 0.08, 0.2)?;
    let nce_after = contrastive.loss(&VIEW_A, &VIEW_B, 0.2);
    let retrieval_after = contrastive.retrieval_accuracy(&VIEW_A, &VIEW_B);
    println!(
        "contrastive InfoNCE {nce_before:.5} -> {nce_after:.5}; paired retrieval {:.1}% -> {:.1}%",
        retrieval_before * 100.0,
        retrieval_after * 100.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_nonfinite_rows_and_temperature_overflow() {
        let mut bad = CLEAN;
        bad[0][0] = f64::NAN;
        assert!(Autoencoder::new().train(&bad, 1, 0.1).is_err());
        assert!(Contrastive::new()
            .train(&bad, &VIEW_B, 1, 0.1, 0.2)
            .is_err());
        assert!(Contrastive::new()
            .train(&VIEW_A, &VIEW_B, 1, 0.1, 1e-320)
            .is_err());
    }

    #[test]
    fn autoencoder_training_reduces_reconstruction_error() {
        let mut ae = Autoencoder::new();
        let before = ae.loss(&CLEAN);
        ae.train(&CLEAN, 500, 0.08).unwrap();
        assert!(ae.loss(&CLEAN) < before * 0.08);
    }
    #[test]
    fn contrastive_training_reduces_stable_infonce() {
        let mut model = Contrastive::new();
        let before = model.loss(&VIEW_A, &VIEW_B, 0.2);
        model.train(&VIEW_A, &VIEW_B, 120, 0.08, 0.2).unwrap();
        assert!(model.loss(&VIEW_A, &VIEW_B, 0.2) < before * 0.8);
    }
    #[test]
    fn autoencoder_encoder_gradient_matches_central_difference() {
        let mut model = Autoencoder::new();
        let analytic = model.encoder_gradient(&CLEAN[0]).0[0][0];
        let h = 1e-5;
        model.encoder[0][0] += h;
        let plus = model.loss(&CLEAN[..1]);
        model.encoder[0][0] -= 2.0 * h;
        let minus = model.loss(&CLEAN[..1]);
        let numeric = (plus - minus) / (2.0 * h);
        assert!((analytic - numeric).abs() < 1e-6 + 1e-4 * numeric.abs());
    }
    #[test]
    fn analytic_infonce_gradient_matches_central_difference() {
        let mut model = Contrastive::new();
        let analytic = model.loss_grad(&VIEW_A, &VIEW_B, 0.2).1[0][0];
        let h = 1e-5;
        model.w[0][0] += h;
        let plus = model.loss(&VIEW_A, &VIEW_B, 0.2);
        model.w[0][0] -= 2.0 * h;
        let minus = model.loss(&VIEW_A, &VIEW_B, 0.2);
        let numeric = (plus - minus) / (2.0 * h);
        assert!(
            (analytic - numeric).abs() < 1e-6 + 1e-4 * numeric.abs(),
            "{analytic} {numeric}"
        );
    }
    #[test]
    fn contrastive_settings_reject_nonfinite_values() {
        let mut model = Contrastive::new();
        assert!(model.train(&VIEW_A, &VIEW_B, 1, f64::NAN, 0.2).is_err());
        assert!(model
            .train(&VIEW_A, &VIEW_B, 1, 0.1, f64::INFINITY)
            .is_err());
    }
    #[test]
    fn bottleneck_has_requested_shape() {
        assert_eq!(Autoencoder::new().encode(&CLEAN[0]).len(), 2);
    }
}
