//! Chapter 51 learner algorithms. Baseline uses reconstruction-only VAE, minimax GAN generator, and no reverse refinement. Implement each complete objective and DDIM transition, then compare samples rather than losses.
include!("common/ch51.rs");
include!("checks/ch51.rs");
pub fn reparameterize(mean: f64, log_variance: f64, epsilon: f64) -> f64 {
    mean + (0.5 * log_variance).exp() * epsilon
}

impl Vae {
    fn loss(&self, data: &[f64], latent_noise: &[f64]) -> Result<f64, &'static str> {
        validate_values(&self.parameters)?;
        validate_values(data)?;
        validate_values(latent_noise)?;
        // Deterministic autoencoder baseline. The full VAE adds a distribution,
        // an expectation over independent noise, and KL regularization.
        let [a, b, _log_variance, d, e] = self.parameters;
        let loss = data
            .iter()
            .map(|&x| {
                let code = a * x + b;
                0.5 * (d * code + e - x).powi(2)
            })
            .sum::<f64>()
            / data.len() as f64;
        loss.is_finite()
            .then_some(loss)
            .ok_or("autoencoder loss became nonfinite")
    }
}

impl Gan {
    fn discriminator_loss(
        discriminator: &[f64; 3],
        generator: &[f64; 2],
        data: &[f64],
        noise: &[f64],
    ) -> Result<f64, &'static str> {
        validate_values(discriminator)?;
        validate_values(generator)?;
        validate_values(data)?;
        validate_values(noise)?;
        if data.len() != noise.len() {
            return Err("GAN data and noise must have equal lengths");
        }
        let loss = data
            .iter()
            .zip(noise)
            .map(|(&real, &z)| {
                let real_logit = Self::discriminator_logit(discriminator, real);
                let fake_logit = Self::discriminator_logit(discriminator, Self::fake(generator, z));
                -log_sigmoid(real_logit) - log_sigmoid(-fake_logit)
            })
            .sum::<f64>()
            / data.len() as f64;
        loss.is_finite()
            .then_some(loss)
            .ok_or("GAN discriminator loss became nonfinite")
    }
}

impl Gan {
    fn generator_loss(
        generator: &[f64; 2],
        discriminator: &[f64; 3],
        noise: &[f64],
    ) -> Result<f64, &'static str> {
        validate_values(generator)?;
        validate_values(discriminator)?;
        validate_values(noise)?;
        let loss = noise
            .iter()
            .map(|&z| {
                log_sigmoid(-Self::discriminator_logit(
                    discriminator,
                    Self::fake(generator, z),
                ))
            })
            .sum::<f64>()
            / noise.len() as f64;
        loss.is_finite()
            .then_some(loss)
            .ok_or("GAN generator loss became nonfinite")
    }
}

impl Diffusion {
    fn loss(&self, data: &[f64], noise: &[f64], alpha_bars: &[f64]) -> Result<f64, &'static str> {
        validate_values(&self.predictor)?;
        validate_values(data)?;
        validate_values(noise)?;
        validate_schedule(alpha_bars)?;
        let mut sum = 0.0;
        let mut count = 0;
        for (time, &alpha_bar) in alpha_bars.iter().enumerate() {
            for &x0 in data {
                for &epsilon in noise {
                    let xt = alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * epsilon;
                    let time_feature = time as f64 / (alpha_bars.len() - 1) as f64;
                    let predicted = self.predictor[0] * xt
                        + self.predictor[1] * time_feature
                        + self.predictor[2];
                    sum += (predicted - epsilon).powi(2);
                    count += 1;
                }
            }
        }
        let loss = sum / count as f64;
        loss.is_finite()
            .then_some(loss)
            .ok_or("diffusion loss became nonfinite")
    }
}

fn ddim_step(xt: f64, time: usize, epsilon: f64, alpha_bars: &[f64]) -> Result<f64, &'static str> {
    validate_schedule(alpha_bars)?;
    if time >= alpha_bars.len() || !xt.is_finite() || !epsilon.is_finite() {
        return Err("DDIM step inputs are invalid");
    }
    // Baseline sampler leaves the initial noise unchanged.
    Ok(xt)
}
