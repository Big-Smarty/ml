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

struct AutoencoderForwardCache {
    code: [f64; LATENT],
    reconstruction: [f64; INPUTS],
}

fn reconstruction_mse(reconstruction: &[f64], target: &[f64]) -> f64 {
    reconstruction
        .iter()
        .zip(target)
        .map(|(prediction, target)| (prediction - target).powi(2))
        .sum::<f64>()
        / target.len() as f64
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
    fn encode(&self, features: &[f64; INPUTS]) -> [f64; LATENT] {
        let mut code = self.encoder_bias;
        for (h, value) in code.iter_mut().enumerate() {
            for (j, &feature) in features.iter().enumerate() {
                *value += self.encoder[h][j] * feature;
            }
            *value = value.tanh();
        }
        code
    }

    fn forward(&self, features: &[f64; INPUTS]) -> AutoencoderForwardCache {
        let code = self.encode(features);
        let mut reconstruction = self.decoder_bias;
        for (j, value) in reconstruction.iter_mut().enumerate() {
            for (h, &latent) in code.iter().enumerate() {
                *value += self.decoder[j][h] * latent;
            }
        }
        AutoencoderForwardCache {
            code,
            reconstruction,
        }
    }

    fn reconstruct(&self, features: &[f64; INPUTS]) -> [f64; INPUTS] {
        self.forward(features).reconstruction
    }

    fn loss(&self, inputs: &[[f64; INPUTS]]) -> f64 {
        inputs
            .iter()
            .map(|features| reconstruction_mse(&self.reconstruct(features), features))
            .sum::<f64>()
            / inputs.len() as f64
    }

    fn encoder_gradient(
        &self,
        features: &[f64; INPUTS],
    ) -> ([[f64; INPUTS]; LATENT], [f64; LATENT]) {
        let cache = self.forward(features);
        let reconstruction_gradient: [f64; INPUTS] =
            std::array::from_fn(|j| 2.0 * (cache.reconstruction[j] - features[j]) / INPUTS as f64);
        let mut gradient = [[0.0; INPUTS]; LATENT];
        let mut bias_gradient = [0.0; LATENT];
        for h in 0..LATENT {
            let code_gradient: f64 = (0..INPUTS)
                .map(|j| self.decoder[j][h] * reconstruction_gradient[j])
                .sum();
            let pre_activation_gradient = code_gradient * (1.0 - cache.code[h] * cache.code[h]);
            bias_gradient[h] = pre_activation_gradient;
            for j in 0..INPUTS {
                gradient[h][j] = pre_activation_gradient * features[j];
            }
        }
        (gradient, bias_gradient)
    }
    fn train(
        &mut self,
        inputs: &[[f64; INPUTS]],
        epochs: usize,
        learning_rate: f64,
    ) -> Result<(), &'static str> {
        if inputs.is_empty()
            || inputs.iter().flatten().any(|feature| !feature.is_finite())
            || epochs == 0
            || learning_rate <= 0.0
            || !learning_rate.is_finite()
        {
            return Err("nonempty inputs, positive epochs and learning rate required");
        }
        for _ in 0..epochs {
            for features in inputs {
                let cache = self.forward(features);
                if cache
                    .reconstruction
                    .iter()
                    .chain(&cache.code)
                    .any(|value| !value.is_finite())
                {
                    return Err("autoencoder overflow; reduce the learning rate");
                }
                let (encoder_gradient, encoder_bias_gradient) = self.encoder_gradient(features);
                let reconstruction_gradient: [f64; INPUTS] = std::array::from_fn(|j| {
                    2.0 * (cache.reconstruction[j] - features[j]) / INPUTS as f64
                });
                for (j, &output_gradient) in reconstruction_gradient.iter().enumerate() {
                    for (h, &latent) in cache.code.iter().enumerate() {
                        self.decoder[j][h] -= learning_rate * output_gradient * latent;
                    }
                    self.decoder_bias[j] -= learning_rate * output_gradient;
                }
                for (h, row) in self.encoder.iter_mut().enumerate() {
                    for (j, weight) in row.iter_mut().enumerate() {
                        *weight -= learning_rate * encoder_gradient[h][j];
                    }
                    self.encoder_bias[h] -= learning_rate * encoder_bias_gradient[h];
                }
                if self
                    .encoder
                    .iter()
                    .flatten()
                    .chain(self.decoder.iter().flatten())
                    .chain(&self.encoder_bias)
                    .chain(&self.decoder_bias)
                    .any(|value| !value.is_finite())
                {
                    return Err("autoencoder update overflow; reduce the learning rate");
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Contrastive {
    weights: [[f64; INPUTS]; LATENT],
}

struct ContrastiveForwardCache {
    raw: [f64; LATENT],
    norm: f64,
    embedding: [f64; LATENT],
}

impl Contrastive {
    fn new() -> Self {
        Self {
            weights: [[0.31, -0.22, 0.17, 0.09], [-0.11, 0.27, 0.08, -0.24]],
        }
    }

    fn forward(&self, features: &[f64; INPUTS]) -> ContrastiveForwardCache {
        let mut raw = [0.0; LATENT];
        for (h, value) in raw.iter_mut().enumerate() {
            for (j, &feature) in features.iter().enumerate() {
                *value += self.weights[h][j] * feature;
            }
        }
        let norm = (raw.iter().map(|v| v * v).sum::<f64>() + 1e-8).sqrt();
        ContrastiveForwardCache {
            raw,
            norm,
            embedding: raw.map(|value| value / norm),
        }
    }

    fn embed(&self, features: &[f64; INPUTS]) -> [f64; LATENT] {
        self.forward(features).embedding
    }
    fn dot(a: [f64; LATENT], b: [f64; LATENT]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
    fn loss(&self, view_a: &[[f64; INPUTS]], view_b: &[[f64; INPUTS]], temperature: f64) -> f64 {
        let embeddings_a: Vec<_> = view_a.iter().map(|features| self.embed(features)).collect();
        let embeddings_b: Vec<_> = view_b.iter().map(|features| self.embed(features)).collect();
        let mut total = 0.0;
        for i in 0..view_a.len() {
            let logits: Vec<_> = embeddings_b
                .iter()
                .map(|&candidate| Self::dot(embeddings_a[i], candidate) / temperature)
                .collect();
            let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            total += (m - logits[i]) + logits.iter().map(|&z| (z - m).exp()).sum::<f64>().ln();
        }
        total / view_a.len() as f64
    }

    fn loss_and_gradient(
        &self,
        view_a: &[[f64; INPUTS]],
        view_b: &[[f64; INPUTS]],
        temperature: f64,
    ) -> (f64, [[f64; INPUTS]; LATENT]) {
        let cache_a: Vec<_> = view_a
            .iter()
            .map(|features| self.forward(features))
            .collect();
        let cache_b: Vec<_> = view_b
            .iter()
            .map(|features| self.forward(features))
            .collect();
        let mut embedding_gradient_a = vec![[0.0; LATENT]; view_a.len()];
        let mut embedding_gradient_b = vec![[0.0; LATENT]; view_b.len()];
        let mut loss = 0.0;
        for i in 0..view_a.len() {
            let logits: Vec<_> = cache_b
                .iter()
                .map(|candidate| Self::dot(cache_a[i].embedding, candidate.embedding) / temperature)
                .collect();
            let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = logits.iter().map(|&z| (z - m).exp()).sum();
            loss += (m - logits[i]) + sum.ln();
            for j in 0..view_b.len() {
                let mut logit_gradient = (logits[j] - m).exp() / sum;
                logit_gradient -= f64::from(i == j);
                logit_gradient /= view_a.len() as f64;
                for h in 0..LATENT {
                    embedding_gradient_a[i][h] +=
                        logit_gradient * cache_b[j].embedding[h] / temperature;
                    embedding_gradient_b[j][h] +=
                        logit_gradient * cache_a[i].embedding[h] / temperature;
                }
            }
        }
        let mut gradient = [[0.0; INPUTS]; LATENT];
        for (views, cache, embedding_gradient) in [
            (view_a, &cache_a, &embedding_gradient_a),
            (view_b, &cache_b, &embedding_gradient_b),
        ] {
            for n in 0..views.len() {
                let dot: f64 = embedding_gradient[n]
                    .iter()
                    .zip(cache[n].raw)
                    .map(|(gradient, raw)| gradient * raw)
                    .sum();
                for h in 0..LATENT {
                    let raw_gradient = embedding_gradient[n][h] / cache[n].norm
                        - cache[n].raw[h] * dot / cache[n].norm.powi(3);
                    for j in 0..INPUTS {
                        gradient[h][j] += raw_gradient * views[n][j];
                    }
                }
            }
        }
        (loss / view_a.len() as f64, gradient)
    }
    fn train(
        &mut self,
        view_a: &[[f64; INPUTS]],
        view_b: &[[f64; INPUTS]],
        epochs: usize,
        learning_rate: f64,
        temperature: f64,
    ) -> Result<(), &'static str> {
        if view_a
            .iter()
            .chain(view_b)
            .flatten()
            .any(|feature| !feature.is_finite())
            || view_a.is_empty()
            || view_a.len() != view_b.len()
            || epochs == 0
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
            || !temperature.is_finite()
            || temperature <= 0.0
        {
            return Err("paired nonempty views and positive settings required");
        }
        for _ in 0..epochs {
            let (loss, gradient) = self.loss_and_gradient(view_a, view_b, temperature);
            if !loss.is_finite() || gradient.iter().flatten().any(|value| !value.is_finite()) {
                return Err(
                    "contrastive arithmetic overflow; increase temperature or reduce the learning rate",
                );
            }
            for (r, row) in self.weights.iter_mut().enumerate() {
                for (c, weight) in row.iter_mut().enumerate() {
                    *weight -= learning_rate * gradient[r][c];
                    if !weight.is_finite() {
                        return Err("contrastive update overflow; reduce the learning rate");
                    }
                }
            }
        }
        Ok(())
    }
    fn retrieval_accuracy(&self, view_a: &[[f64; INPUTS]], view_b: &[[f64; INPUTS]]) -> f64 {
        let embeddings_b: Vec<_> = view_b.iter().map(|features| self.embed(features)).collect();
        let correct = view_a
            .iter()
            .enumerate()
            .filter(|(i, features)| {
                let embedding = self.embed(features);
                let best = (0..view_b.len())
                    .max_by(|&p, &q| {
                        Self::dot(embedding, embeddings_b[p])
                            .total_cmp(&Self::dot(embedding, embeddings_b[q]))
                    })
                    .unwrap();
                best == *i
            })
            .count();
        correct as f64 / view_a.len() as f64
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut autoencoder = Autoencoder::new();
    let reconstruction_before = autoencoder.loss(&CLEAN);
    autoencoder.train(&CLEAN, 500, 0.08)?;
    let reconstruction_after = autoencoder.loss(&CLEAN);
    let code = autoencoder.encode(&CLEAN[0]);
    println!(
        "autoencoder reconstruction MSE {reconstruction_before:.5} -> {reconstruction_after:.5}; first code [{:.3}, {:.3}]",
        code[0], code[1]
    );

    let mut contrastive = Contrastive::new();
    let infonce_before = contrastive.loss(&VIEW_A, &VIEW_B, 0.2);
    let retrieval_before = contrastive.retrieval_accuracy(&VIEW_A, &VIEW_B);
    contrastive.train(&VIEW_A, &VIEW_B, 120, 0.08, 0.2)?;
    let infonce_after = contrastive.loss(&VIEW_A, &VIEW_B, 0.2);
    let retrieval_after = contrastive.retrieval_accuracy(&VIEW_A, &VIEW_B);
    println!(
        "contrastive InfoNCE {infonce_before:.5} -> {infonce_after:.5}; paired retrieval {:.1}% -> {:.1}%",
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
        let mut model = Autoencoder::new();
        let before = model.loss(&CLEAN);
        model.train(&CLEAN, 500, 0.08).unwrap();
        assert!(model.loss(&CLEAN) < before * 0.08);
    }
    #[test]
    fn contrastive_training_reduces_stable_infonce() {
        let mut model = Contrastive::new();
        let before = model.loss(&VIEW_A, &VIEW_B, 0.2);
        model.train(&VIEW_A, &VIEW_B, 120, 0.08, 0.2).unwrap();
        assert!(model.loss(&VIEW_A, &VIEW_B, 0.2) < before * 0.8);
    }
    #[test]
    fn autoencoder_reconstruction_gradient_matches_central_difference() {
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
        let analytic = model.loss_and_gradient(&VIEW_A, &VIEW_B, 0.2).1[0][0];
        let h = 1e-5;
        model.weights[0][0] += h;
        let plus = model.loss(&VIEW_A, &VIEW_B, 0.2);
        model.weights[0][0] -= 2.0 * h;
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

    #[test]
    fn objectives_use_their_documented_reductions() {
        let autoencoder = Autoencoder {
            encoder: [[0.0; INPUTS]; LATENT],
            encoder_bias: [0.0; LATENT],
            decoder: [[0.0; LATENT]; INPUTS],
            decoder_bias: [0.0; INPUTS],
        };
        let inputs = [[1.0, 3.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]];
        assert_eq!(autoencoder.loss(&inputs), 14.0 / 8.0);

        let contrastive = Contrastive {
            weights: [[0.0; INPUTS]; LATENT],
        };
        assert!((contrastive.loss(&VIEW_A[..3], &VIEW_B[..3], 0.2) - 3.0_f64.ln()).abs() < 1e-12);
    }
}
