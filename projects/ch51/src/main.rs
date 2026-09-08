//! Three tiny trainable generative models with deterministic teaching fixtures.
use std::f64::consts::TAU;

const DATA: [f64; 6] = [-1.4, -1.0, -0.7, 0.7, 1.0, 1.4];
const NOISE: [f64; 6] = [-1.2, -0.7, -0.2, 0.2, 0.7, 1.2];

fn validate_values(values: &[f64]) -> Result<(), &'static str> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("values must be nonempty and finite");
    }
    Ok(())
}

fn validate_schedule(alpha_bars: &[f64]) -> Result<(), &'static str> {
    if alpha_bars.len() < 2
        || alpha_bars
            .iter()
            .any(|&alpha_bar| !alpha_bar.is_finite() || alpha_bar <= 0.0 || alpha_bar > 1.0)
        || alpha_bars.windows(2).any(|pair| pair[1] > pair[0])
    {
        return Err("schedule must contain at least two finite, nonincreasing values in (0, 1]");
    }
    Ok(())
}

fn validate_learning_rate(learning_rate: f64) -> Result<(), &'static str> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("learning rate must be positive and finite");
    }
    Ok(())
}

fn numerical_gradient<const N: usize>(
    parameters: &[f64; N],
    loss: impl Fn(&[f64; N]) -> Result<f64, &'static str>,
) -> Result<[f64; N], &'static str> {
    // ponytail: two loss evaluations per parameter; replace with backpropagation for larger models.
    validate_values(parameters)?;
    let mut gradient = [0.0; N];
    let h = 1e-5;
    for i in 0..N {
        let mut plus = *parameters;
        let mut minus = *parameters;
        plus[i] += h;
        minus[i] -= h;
        gradient[i] = (loss(&plus)? - loss(&minus)?) / (2.0 * h);
    }
    validate_values(&gradient)?;
    Ok(gradient)
}

fn apply_sgd<const N: usize>(
    parameters: &mut [f64; N],
    gradient: [f64; N],
    learning_rate: f64,
) -> Result<(), &'static str> {
    validate_learning_rate(learning_rate)?;
    let next = std::array::from_fn(|i| parameters[i] - learning_rate * gradient[i]);
    validate_values(&next)?;
    *parameters = next;
    Ok(())
}

fn reparameterize(mean: f64, log_variance: f64, epsilon: f64) -> f64 {
    mean + (0.5 * log_variance).exp() * epsilon
}

#[derive(Clone, Copy, Debug)]
struct Vae {
    parameters: [f64; 5],
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

    fn train(
        mut self,
        data: &[f64],
        latent_noise: &[f64],
        steps: usize,
        learning_rate: f64,
    ) -> Result<Self, &'static str> {
        validate_learning_rate(learning_rate)?;
        self.loss(data, latent_noise)?;
        for _ in 0..steps {
            let gradient = numerical_gradient(&self.parameters, |parameters| {
                Self {
                    parameters: *parameters,
                }
                .loss(data, latent_noise)
            })?;
            apply_sgd(&mut self.parameters, gradient, learning_rate)?;
        }
        Ok(self)
    }

    fn decode(&self, z: f64) -> f64 {
        self.parameters[3] * z + self.parameters[4]
    }
}

fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

#[derive(Clone, Copy, Debug)]
struct Gan {
    generator: [f64; 2],
    discriminator: [f64; 3],
}

impl Gan {
    fn fake(generator: &[f64; 2], z: f64) -> f64 {
        generator[0] * z + generator[1]
    }

    fn discriminator_logit(discriminator: &[f64; 3], x: f64) -> f64 {
        discriminator[0] * x + discriminator[1] * x.powi(2) + discriminator[2]
    }

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

    fn train(
        mut self,
        data: &[f64],
        noise: &[f64],
        rounds: usize,
        learning_rate: f64,
    ) -> Result<Self, &'static str> {
        validate_learning_rate(learning_rate)?;
        Self::discriminator_loss(&self.discriminator, &self.generator, data, noise)?;
        for _ in 0..rounds {
            let discriminator_gradient =
                numerical_gradient(&self.discriminator, |discriminator| {
                    Self::discriminator_loss(discriminator, &self.generator, data, noise)
                })?;
            apply_sgd(
                &mut self.discriminator,
                discriminator_gradient,
                learning_rate,
            )?;
            let generator_gradient = numerical_gradient(&self.generator, |generator| {
                Self::generator_loss(generator, &self.discriminator, noise)
            })?;
            apply_sgd(&mut self.generator, generator_gradient, learning_rate)?;
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug)]
struct Diffusion {
    predictor: [f64; 3],
}

const ALPHA_BARS: [f64; 4] = [0.95, 0.6, 0.15, 0.02];

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

    fn train(
        mut self,
        data: &[f64],
        noise: &[f64],
        alpha_bars: &[f64],
        steps: usize,
        learning_rate: f64,
    ) -> Result<Self, &'static str> {
        validate_learning_rate(learning_rate)?;
        self.loss(data, noise, alpha_bars)?;
        for _ in 0..steps {
            let gradient = numerical_gradient(&self.predictor, |predictor| {
                Self {
                    predictor: *predictor,
                }
                .loss(data, noise, alpha_bars)
            })?;
            apply_sgd(&mut self.predictor, gradient, learning_rate)?;
        }
        Ok(self)
    }

    fn sample(&self, initial_noise: f64, alpha_bars: &[f64]) -> Result<f64, &'static str> {
        if !initial_noise.is_finite() {
            return Err("initial noise must be finite");
        }
        validate_schedule(alpha_bars)?;
        let mut xt = initial_noise;
        for time in (0..alpha_bars.len()).rev() {
            let time_feature = time as f64 / (alpha_bars.len() - 1) as f64;
            let epsilon =
                self.predictor[0] * xt + self.predictor[1] * time_feature + self.predictor[2];
            xt = ddim_step(xt, time, epsilon, alpha_bars)?;
        }
        Ok(xt)
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

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (TAU * self.uniform()).cos()
    }
}

fn report_samples(name: &str, values: &[f64]) {
    assert!(!values.is_empty() && values.iter().all(|x| x.is_finite()));
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let negative = values.iter().filter(|&&x| x < 0.0).count();
    let central = values.iter().filter(|&&x| x.abs() < 0.5).count();
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    // Nearest-rank quartiles: sorted[ceil(p*N)-1].
    let quartiles = [1, 2, 3].map(|i| sorted[(i * sorted.len()).div_ceil(4) - 1]);
    println!("{name}: n={}, mean={mean:.3}, variance={variance:.3}, negative={negative}, |x|<0.5={central}, quartiles={quartiles:.3?}", values.len());
}

fn main() -> Result<(), &'static str> {
    let vae_initial = Vae {
        parameters: [0.4, 0.0, -0.5, 0.4, 0.0],
    };
    let vae = vae_initial.train(&DATA, &NOISE, 500, 0.03)?;
    let gan_initial = Gan {
        generator: [0.25, 0.3],
        discriminator: [0.2, 0.15, 0.0],
    };
    let gan = gan_initial.train(&DATA, &NOISE, 1_500, 0.02)?;
    let diffusion_initial = Diffusion {
        predictor: [0.0; 3],
    };
    let diffusion = diffusion_initial.train(&DATA, &NOISE, &ALPHA_BARS, 500, 0.03)?;
    let mut rng = Rng(51);
    println!(
        "beta-VAE objective: {:.4} -> {:.4}; decoder means: {:.3}, {:.3}",
        vae_initial.loss(&DATA, &NOISE)?,
        vae.loss(&DATA, &NOISE)?,
        vae.decode(rng.normal()),
        vae.decode(rng.normal())
    );
    println!(
        "GAN generator: x={:.3}z{:+.3}; discriminator loss {:.4}",
        gan.generator[0],
        gan.generator[1],
        Gan::discriminator_loss(&gan.discriminator, &gan.generator, &DATA, &NOISE)?
    );
    println!(
        "diffusion noise MSE: {:.4} -> {:.4}; reverse sample {:.3}",
        diffusion_initial.loss(&DATA, &NOISE, &ALPHA_BARS)?,
        diffusion.loss(&DATA, &NOISE, &ALPHA_BARS)?,
        diffusion.sample(rng.normal(), &ALPHA_BARS)?
    );
    // Common latent draws isolate model differences from sampling variation.
    let mut comparison_rng = Rng(51);
    let noise: Vec<_> = (0..512).map(|_| comparison_rng.normal()).collect();
    let vae_means: Vec<_> = noise.iter().map(|&z| vae.decode(z)).collect();
    let vae_samples: Vec<_> = vae_means
        .iter()
        .map(|&mean| mean + comparison_rng.normal())
        .collect();
    let gan_samples: Vec<_> = noise
        .iter()
        .map(|&z| Gan::fake(&gan.generator, z))
        .collect();
    let diffusion_samples: Vec<_> = noise
        .iter()
        .map(|&z| diffusion.sample(z, &ALPHA_BARS))
        .collect::<Result<_, _>>()?;
    report_samples("data fixture", &DATA);
    report_samples("VAE decoder means", &vae_means);
    report_samples("VAE observations (unit decoder variance)", &vae_samples);
    report_samples("GAN", &gan_samples);
    report_samples("diffusion", &diffusion_samples);
    println!("All three models were trained here; affine Gaussian sampling cannot represent these two modes.");
    Ok(())
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
