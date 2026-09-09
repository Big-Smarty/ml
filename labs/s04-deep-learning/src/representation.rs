//! Two real representation-learning experiments: an autoencoder and InfoNCE training.
pub(crate) const INPUTS: usize = 4;
pub(crate) const LATENT: usize = 2;

type AutoencoderGradient =
    fn(&Autoencoder, &[f64; INPUTS]) -> ([[f64; INPUTS]; LATENT], [f64; LATENT]);
type ContrastiveGradient =
    fn(&Contrastive, &[[f64; INPUTS]], &[[f64; INPUTS]], f64) -> (f64, [[f64; INPUTS]; LATENT]);
#[derive(Clone, Copy)]
pub(crate) struct Core {
    pub autoencoder_gradient: AutoencoderGradient,
    pub contrastive_gradient: ContrastiveGradient,
    pub nearest: fn(&[f64; LATENT], &[[f64; LATENT]]) -> usize,
}
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
pub(crate) struct Autoencoder {
    core: Core,
    pub(crate) encoder: [[f64; INPUTS]; LATENT],
    pub(crate) encoder_bias: [f64; LATENT],
    pub(crate) decoder: [[f64; LATENT]; INPUTS],
    pub(crate) decoder_bias: [f64; INPUTS],
}

pub(crate) struct AutoencoderForwardCache {
    pub(crate) code: [f64; LATENT],
    pub(crate) reconstruction: [f64; INPUTS],
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
    fn new(core: Core) -> Self {
        Self {
            core,
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

    pub(crate) fn forward(&self, features: &[f64; INPUTS]) -> AutoencoderForwardCache {
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
        (self.core.autoencoder_gradient)(self, features)
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
pub(crate) struct Contrastive {
    core: Core,
    pub(crate) weights: [[f64; INPUTS]; LATENT],
}

pub(crate) struct ContrastiveForwardCache {
    pub(crate) raw: [f64; LATENT],
    pub(crate) norm: f64,
    pub(crate) embedding: [f64; LATENT],
}

impl Contrastive {
    fn new(core: Core) -> Self {
        Self {
            core,
            weights: [[0.31, -0.22, 0.17, 0.09], [-0.11, 0.27, 0.08, -0.24]],
        }
    }

    pub(crate) fn forward(&self, features: &[f64; INPUTS]) -> ContrastiveForwardCache {
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
    pub(crate) fn dot(a: [f64; LATENT], b: [f64; LATENT]) -> f64 {
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
        (self.core.contrastive_gradient)(self, view_a, view_b, temperature)
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
                let best = (self.core.nearest)(&embedding, &embeddings_b);
                best == *i
            })
            .count();
        correct as f64 / view_a.len() as f64
    }
}

pub(crate) fn run(core: Core, args: &[String]) -> Result<(), String> {
    if let [flag, stage] = args {
        if flag == "--checkpoint" {
            return check_stage(core, stage);
        }
    }
    if !args.is_empty() {
        return Err("usage: 22".into());
    }
    println!(
        "first clean input {:?}; pair A {:?}, B {:?}; shapes4->2->4 and4->2",
        CLEAN[0], VIEW_A[0], VIEW_B[0]
    );
    println!(
        "AE500 epochs, contrastive120 epochs; learning_rate0.08; temperature0.2; six rows/pairs"
    );
    let mut autoencoder = Autoencoder::new(core);
    let reconstruction_before = autoencoder.loss(&CLEAN);
    autoencoder.train(&CLEAN, 500, 0.08)?;
    let reconstruction_after = autoencoder.loss(&CLEAN);
    let code = autoencoder.encode(&CLEAN[0]);
    println!(
        "autoencoder reconstruction MSE {reconstruction_before:.5} -> {reconstruction_after:.5}; first code [{:.3}, {:.3}]",
        code[0], code[1]
    );

    let mut contrastive = Contrastive::new(core);
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

pub(crate) fn check(core: Core) -> Result<(), String> {
    check_stage(core, "all")
}
fn check_stage(core: Core, stage: &str) -> Result<(), String> {
    if !["autoencoder", "contrastive", "neighbors", "training", "all"].contains(&stage) {
        return Err(format!("unknown chapter22 checkpoint {stage}"));
    }

    let input = [0.75, 0.65, 0.35, 0.25];
    let model = Autoencoder::new(core);
    let analytic = model.encoder_gradient(&input);
    for (h, j) in [(0, 0), (1, 2), (1, 3)] {
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus.encoder[h][j] += 1e-5;
        minus.encoder[h][j] -= 1e-5;
        crate::close(
            "autoencoder encoder derivative",
            analytic.0[h][j],
            (plus.loss(&[input]) - minus.loss(&[input])) / 2e-5,
        )?;
    }
    if stage == "autoencoder" {
        println!("PASS autoencoder checkpoint");
        return Ok(());
    }
    let model = Contrastive::new(core);
    let a = [
        [0.7, 0.2, 0.1, 0.4],
        [0.1, 0.6, 0.9, 0.2],
        [0.3, 0.4, 0.2, 0.8],
    ];
    let b = [
        [0.8, 0.1, 0.2, 0.3],
        [0.2, 0.5, 0.8, 0.1],
        [0.4, 0.3, 0.1, 0.7],
    ];
    let (loss, gradient) = model.loss_and_gradient(&a, &b, 0.37);
    crate::close("InfoNCE mean over anchors", loss, model.loss(&a, &b, 0.37))?;
    for (h, j) in [(0, 0), (0, 3), (1, 1)] {
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus.weights[h][j] += 1e-5;
        minus.weights[h][j] -= 1e-5;
        crate::close(
            "shared encoder: both paired branches",
            gradient[h][j],
            (plus.loss(&a, &b, 0.37) - minus.loss(&a, &b, 0.37)) / 2e-5,
        )?;
    }
    if stage == "contrastive" {
        println!("PASS contrastive checkpoint");
        return Ok(());
    }
    let query = [1.0, 0.0];
    let candidates = [[4.0, 0.0], [0.8, 0.6], [-1.0, 0.0]];
    let nearest = (core.nearest)(&query, &candidates);
    if nearest != 0 {
        return Err(format!(
            "GOAL_NOT_MET: cosine neighbor should ignore length: expected candidate0, got {nearest}"
        ));
    }
    if (core.nearest)(&[0.0, 1.0], &[[0.0, 0.0], [0.0, 2.0], [1.0, 0.0]]) != 1 {
        return Err("GOAL_NOT_MET: cosine neighbor: zero vectors need a finite denominator".into());
    }
    if stage == "neighbors" {
        println!("PASS cosine neighborhood on unequal lengths and zero candidate");
        return Ok(());
    }
    let mut ae = Autoencoder::new(core);
    let before = ae.loss(&CLEAN);
    ae.train(&CLEAN, 500, 0.08)?;
    let mut contrastive = Contrastive::new(core);
    let before_c = contrastive.loss(&VIEW_A, &VIEW_B, 0.2);
    contrastive.train(&VIEW_A, &VIEW_B, 120, 0.08, 0.2)?;
    if ae.loss(&CLEAN) >= before * 0.08 || contrastive.loss(&VIEW_A, &VIEW_B, 0.2) >= before_c * 0.8
    {
        return Err(
            "GOAL_NOT_MET: objectives did not decrease by the declared fixture factors".into(),
        );
    }
    println!("PASS AE gradients, asymmetric three-pair shared gradients, both training objectives; unfamiliar reconstruction MSE {:.6}",ae.loss(&[input]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_nonfinite_rows_and_temperature_overflow() {
        let mut bad = CLEAN;
        bad[0][0] = f64::NAN;
        assert!(Autoencoder::new(crate::solutions::ch22::CORE)
            .train(&bad, 1, 0.1)
            .is_err());
        assert!(Contrastive::new(crate::solutions::ch22::CORE)
            .train(&bad, &VIEW_B, 1, 0.1, 0.2)
            .is_err());
        assert!(Contrastive::new(crate::solutions::ch22::CORE)
            .train(&VIEW_A, &VIEW_B, 1, 0.1, 1e-320)
            .is_err());
    }

    #[test]
    fn autoencoder_training_reduces_reconstruction_error() {
        let mut model = Autoencoder::new(crate::solutions::ch22::CORE);
        let before = model.loss(&CLEAN);
        model.train(&CLEAN, 500, 0.08).unwrap();
        assert!(model.loss(&CLEAN) < before * 0.08);
    }
    #[test]
    fn contrastive_training_reduces_stable_infonce() {
        let mut model = Contrastive::new(crate::solutions::ch22::CORE);
        let before = model.loss(&VIEW_A, &VIEW_B, 0.2);
        model.train(&VIEW_A, &VIEW_B, 120, 0.08, 0.2).unwrap();
        assert!(model.loss(&VIEW_A, &VIEW_B, 0.2) < before * 0.8);
    }
    #[test]
    fn autoencoder_reconstruction_gradient_matches_central_difference() {
        let mut model = Autoencoder::new(crate::solutions::ch22::CORE);
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
        let mut model = Contrastive::new(crate::solutions::ch22::CORE);
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
        let mut model = Contrastive::new(crate::solutions::ch22::CORE);
        assert!(model.train(&VIEW_A, &VIEW_B, 1, f64::NAN, 0.2).is_err());
        assert!(model
            .train(&VIEW_A, &VIEW_B, 1, 0.1, f64::INFINITY)
            .is_err());
    }
    #[test]
    fn bottleneck_has_requested_shape() {
        assert_eq!(
            Autoencoder::new(crate::solutions::ch22::CORE)
                .encode(&CLEAN[0])
                .len(),
            2
        );
    }

    #[test]
    fn objectives_use_their_documented_reductions() {
        let autoencoder = Autoencoder {
            core: crate::solutions::ch22::CORE,
            encoder: [[0.0; INPUTS]; LATENT],
            encoder_bias: [0.0; LATENT],
            decoder: [[0.0; LATENT]; INPUTS],
            decoder_bias: [0.0; INPUTS],
        };
        let inputs = [[1.0, 3.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]];
        assert_eq!(autoencoder.loss(&inputs), 14.0 / 8.0);

        let contrastive = Contrastive {
            core: crate::solutions::ch22::CORE,
            weights: [[0.0; INPUTS]; LATENT],
        };
        assert!((contrastive.loss(&VIEW_A[..3], &VIEW_B[..3], 0.2) - 3.0_f64.ln()).abs() < 1e-12);
    }
}
