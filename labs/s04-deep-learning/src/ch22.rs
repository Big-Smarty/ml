//! Baselines: fixed random encoder + trained decoder, and anchor-only contrastive update.
//! Open the fixed encoder to learning; then add candidate-path derivatives to the shared encoder.
//! All views and training/reporting plumbing are supplied in representation.rs.
use crate::representation::{self, Autoencoder, Contrastive, Core, INPUTS, LATENT};
pub(crate) const CORE: Core = Core {
    autoencoder_gradient,
    contrastive_gradient,
    nearest,
};

pub(crate) fn autoencoder_gradient(
    _model: &Autoencoder,
    _features: &[f64; INPUTS],
) -> ([[f64; INPUTS]; LATENT], [f64; LATENT]) {
    // A fixed random feature map is a useful baseline: the decoder still learns.
    ([[0.0; INPUTS]; LATENT], [0.0; LATENT])
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
            for (h, gradient) in embedding_gradient_a[i].iter_mut().enumerate() {
                *gradient += logit_gradient * cache_b[j].embedding[h] / temperature;
            }
        }
    }
    let mut gradient = [[0.0; INPUTS]; LATENT];
    for (views, cache, embedding_gradient) in [(view_a, &cache_a, &embedding_gradient_a)] {
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
    // Baseline Euclidean distance also considers vector lengths.
    let mut best = 0;
    let mut best_distance = f64::INFINITY;
    for (j, candidate) in candidates.iter().enumerate() {
        let distance = query
            .iter()
            .zip(candidate)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>();
        if distance < best_distance {
            best = j;
            best_distance = distance;
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
