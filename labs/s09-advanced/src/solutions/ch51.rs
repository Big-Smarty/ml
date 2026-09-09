//! Worked solution: Average decoder reconstruction over fixed latent noise and add beta-weighted Gaussian KL once per observation. The discriminator and non-saturating generator use distinct stable log-sigmoid objectives with the other parameter block fixed. DDIM first estimates clean data and then forms the previous state using the same predicted noise.
//! Compare intermediate quantities with the lesson before using the final goal check.
//! Worked solution for chapter 51. Read the comments and lesson explanations before comparing.
include!("../common/ch51.rs");
include!("../checks/ch51.rs");
fn reparameterize(mean: f64, log_variance: f64, epsilon: f64) -> f64 {
    mean + (0.5 * log_variance).exp() * epsilon
}

impl Vae {
    fn loss(&self, data: &[f64], latent_noise: &[f64]) -> Result<f64, &'static str> {
        validate_values(&self.parameters)?;
        validate_values(data)?;
        validate_values(latent_noise)?;
        let [encoder_weight, encoder_bias, log_variance, decoder_weight, decoder_bias] =
            self.parameters;
        let loss = data
            .iter()
            .flat_map(|&x| latent_noise.iter().map(move |&epsilon| (x, epsilon)))
            .map(|(x, epsilon)| {
                let mean = encoder_weight * x + encoder_bias;
                let variance = log_variance.exp();
                let z = reparameterize(mean, log_variance, epsilon);
                let reconstruction = decoder_weight * z + decoder_bias;
                let reconstruction_loss = 0.5 * (reconstruction - x).powi(2);
                let kl = 0.5 * (mean.powi(2) + variance - 1.0 - log_variance);
                reconstruction_loss + 0.1 * kl
            })
            .sum::<f64>()
            / (data.len() * latent_noise.len()) as f64;
        loss.is_finite()
            .then_some(loss)
            .ok_or("VAE loss became nonfinite")
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
                -log_sigmoid(Self::discriminator_logit(
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
    let alpha_bar = alpha_bars[time];
    let x0_estimate = (xt - (1.0 - alpha_bar).sqrt() * epsilon) / alpha_bar.sqrt();
    let previous_alpha_bar = if time == 0 { 1.0 } else { alpha_bars[time - 1] };
    let previous =
        previous_alpha_bar.sqrt() * x0_estimate + (1.0 - previous_alpha_bar).sqrt() * epsilon;
    previous
        .is_finite()
        .then_some(previous)
        .ok_or("DDIM step became nonfinite")
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vae_and_diffusion_objectives_fall() {
        let vae = Vae {
            parameters: [0.4, 0.0, -0.5, 0.4, 0.0],
        };
        let trained_vae = vae.train(&DATA, &NOISE, 500, 0.03).unwrap();
        assert!(trained_vae.loss(&DATA, &NOISE).unwrap() < vae.loss(&DATA, &NOISE).unwrap() * 0.7);
        let diffusion = Diffusion {
            predictor: [0.0; 3],
        };
        assert!(
            diffusion
                .train(&DATA, &NOISE, &ALPHA_BARS, 500, 0.03)
                .unwrap()
                .loss(&DATA, &NOISE, &ALPHA_BARS)
                .unwrap()
                < diffusion.loss(&DATA, &NOISE, &ALPHA_BARS).unwrap() * 0.8
        );
    }

    #[test]
    fn gan_updates_both_networks_and_outputs_finite_samples() {
        let initial = Gan {
            generator: [0.25, 0.3],
            discriminator: [0.2, 0.15, 0.0],
        };
        let trained = initial.train(&DATA, &NOISE, 200, 0.02).unwrap();
        assert_ne!(trained.generator, initial.generator);
        assert_ne!(trained.discriminator, initial.discriminator);
        assert!(NOISE
            .iter()
            .all(|&z| Gan::fake(&trained.generator, z).is_finite()));
    }

    #[test]
    fn finite_difference_matches_quadratic_and_vae_reparameterization() {
        let gradient = numerical_gradient(&[0.3, -0.7], |parameters| {
            Ok(parameters[0].powi(2) + 3.0 * parameters[1].powi(2))
        })
        .unwrap();
        assert!((gradient[0] - 0.6).abs() < 1e-8);
        assert!((gradient[1] + 4.2).abs() < 1e-8);
        // Independent closed-form reconstruction expectation for this symmetric noise grid.
        let p: [f64; 5] = [0.4, 0.1, -0.5, 0.7, -0.2];
        let noise_variance = NOISE.iter().map(|e| e * e).sum::<f64>() / NOISE.len() as f64;
        let expected = DATA
            .iter()
            .map(|&x| {
                let mean = p[0] * x + p[1];
                0.5 * ((p[3] * mean + p[4] - x).powi(2)
                    + p[3].powi(2) * p[2].exp() * noise_variance)
                    + 0.05 * (mean.powi(2) + p[2].exp() - 1.0 - p[2])
            })
            .sum::<f64>()
            / DATA.len() as f64;
        let vae = Vae { parameters: p };
        assert!((vae.loss(&DATA, &NOISE).unwrap() - expected).abs() < 1e-12);

        let zero_vae = Vae {
            parameters: [0.0; 5],
        };
        let alternate_data_loss = zero_vae.loss(&[2.0], &NOISE).unwrap();
        assert!((alternate_data_loss - 2.0).abs() < 1e-12);
        assert!(vae.loss(&[], &NOISE).is_err());
        assert!(vae.loss(&[f64::NAN], &NOISE).is_err());
        assert!(Gan::discriminator_loss(&[0.0; 3], &[0.0; 2], &[0.0, 1.0], &[0.0]).is_err());
        assert!(Diffusion {
            predictor: [0.0; 3]
        }
        .loss(&DATA, &NOISE, &[0.95])
        .is_err());
    }

    #[test]
    fn perfect_noise_prediction_recovers_the_previous_state() {
        let x0 = 0.7;
        let epsilon = -0.4;
        for time in 0..ALPHA_BARS.len() {
            let alpha_bar = ALPHA_BARS[time];
            let xt = alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * epsilon;
            let previous_alpha_bar = if time == 0 { 1.0 } else { ALPHA_BARS[time - 1] };
            let expected =
                previous_alpha_bar.sqrt() * x0 + (1.0 - previous_alpha_bar).sqrt() * epsilon;
            assert!((ddim_step(xt, time, epsilon, &ALPHA_BARS).unwrap() - expected).abs() < 1e-12);
        }
    }
}
