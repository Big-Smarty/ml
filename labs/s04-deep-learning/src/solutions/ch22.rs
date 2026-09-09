//! AE gradient: output MSE -> old decoder -> tanh -> encoder.
//! InfoNCE: accumulate anchor AND candidate derivatives, then differentiate normalization.
use crate::representation::{self, Autoencoder, Contrastive, Core, INPUTS, LATENT};
pub(crate) const CORE: Core = Core {
    autoencoder_gradient,
    contrastive_gradient,
    nearest,
};
pub(crate) fn autoencoder_gradient(
    model: &Autoencoder,
    features: &[f64; INPUTS],
) -> ([[f64; INPUTS]; LATENT], [f64; LATENT]) {
    let cache = model.forward(features);
    let reconstruction_gradient: [f64; INPUTS] =
        std::array::from_fn(|j| 2.0 * (cache.reconstruction[j] - features[j]) / INPUTS as f64);
    let mut gradient = [[0.0; INPUTS]; LATENT];
    let mut bias_gradient = [0.0; LATENT];
    for h in 0..LATENT {
        let code_gradient: f64 = (0..INPUTS)
            .map(|j| model.decoder[j][h] * reconstruction_gradient[j])
            .sum();
        let pre_activation_gradient = code_gradient * (1.0 - cache.code[h] * cache.code[h]);
        bias_gradient[h] = pre_activation_gradient;
        for j in 0..INPUTS {
            gradient[h][j] = pre_activation_gradient * features[j];
        }
    }
    (gradient, bias_gradient)
}
pub(crate) fn contrastive_gradient(
    model: &Contrastive,
    view_a: &[[f64; INPUTS]],
    view_b: &[[f64; INPUTS]],
    temperature: f64,
) -> (f64, [[f64; INPUTS]; LATENT]) {
    let cache_a: Vec<_> = view_a
        .iter()
        .map(|features| model.forward(features))
        .collect();
    let cache_b: Vec<_> = view_b
        .iter()
        .map(|features| model.forward(features))
        .collect();
    let mut embedding_gradient_a = vec![[0.0; LATENT]; view_a.len()];
    let mut embedding_gradient_b = vec![[0.0; LATENT]; view_b.len()];
    let mut loss = 0.0;
    for i in 0..view_a.len() {
        let logits: Vec<_> = cache_b
            .iter()
            .map(|candidate| {
                Contrastive::dot(cache_a[i].embedding, candidate.embedding) / temperature
            })
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
pub(crate) fn nearest(query: &[f64; LATENT], candidates: &[[f64; LATENT]]) -> usize {
    let norm = |x: &[f64; LATENT]| (x.iter().map(|v| v * v).sum::<f64>() + 1e-8).sqrt();
    let query_norm = norm(query);
    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (j, candidate) in candidates.iter().enumerate() {
        let similarity = query.iter().zip(candidate).map(|(a, b)| a * b).sum::<f64>()
            / (query_norm * norm(candidate));
        if similarity > best_score {
            best = j;
            best_score = similarity;
        }
    }
    best
}
pub fn run(args: &[String]) -> Result<(), String> {
    representation::run(CORE, args)
}
pub fn check() -> Result<(), String> {
    representation::check(CORE)
}
