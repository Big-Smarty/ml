//! Three tiny trainable generative models with deterministic teaching fixtures.
use std::f64::consts::TAU;

const DATA: [f64; 6] = [-1.4, -1.0, -0.7, 0.7, 1.0, 1.4];
const NOISE: [f64; 6] = [-1.2, -0.7, -0.2, 0.2, 0.7, 1.2];

fn finite_gradient<const N: usize>(params: &[f64; N], loss: impl Fn(&[f64; N]) -> f64) -> [f64; N] {
    // ponytail: two loss evaluations per parameter; replace with backpropagation for larger models.
    let mut gradient = [0.0; N];
    let h = 1e-5;
    for i in 0..N {
        let mut plus = *params;
        let mut minus = *params;
        plus[i] += h;
        minus[i] -= h;
        gradient[i] = (loss(&plus) - loss(&minus)) / (2.0 * h);
    }
    gradient
}

fn update<const N: usize>(params: &mut [f64; N], gradient: [f64; N], rate: f64) {
    for (parameter, slope) in params.iter_mut().zip(gradient) {
        *parameter -= rate * slope;
    }
}

#[derive(Clone, Copy, Debug)]
struct Vae {
    p: [f64; 5],
}

impl Vae {
    fn loss_with(params: &[f64; 5]) -> f64 {
        let [encoder_w, encoder_b, log_variance, decoder_w, decoder_b] = *params;
        DATA.iter()
            .flat_map(|&x| NOISE.into_iter().map(move |epsilon| (x, epsilon)))
            .map(|(x, epsilon)| {
                let mean = encoder_w * x + encoder_b;
                let variance = log_variance.exp();
                let z = mean + variance.sqrt() * epsilon;
                let reconstruction = decoder_w * z + decoder_b;
                let reconstruction_loss = 0.5 * (reconstruction - x).powi(2);
                let kl = 0.5 * (mean.powi(2) + variance - 1.0 - log_variance);
                reconstruction_loss + 0.1 * kl
            })
            .sum::<f64>()
            / (DATA.len() * NOISE.len()) as f64
    }

    fn train(mut self, steps: usize) -> Self {
        for _ in 0..steps {
            let gradient = finite_gradient(&self.p, Self::loss_with);
            update(&mut self.p, gradient, 0.03);
        }
        self
    }

    fn decode(self, z: f64) -> f64 {
        self.p[3] * z + self.p[4]
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

    fn discriminator_loss(discriminator: &[f64; 3], generator: &[f64; 2]) -> f64 {
        DATA.iter()
            .zip(NOISE)
            .map(|(&real, z)| {
                let real_logit = Self::discriminator_logit(discriminator, real);
                let fake_logit = Self::discriminator_logit(discriminator, Self::fake(generator, z));
                -log_sigmoid(real_logit) - log_sigmoid(-fake_logit)
            })
            .sum::<f64>()
            / DATA.len() as f64
    }

    fn generator_loss(generator: &[f64; 2], discriminator: &[f64; 3]) -> f64 {
        NOISE
            .iter()
            .map(|&z| {
                -log_sigmoid(Self::discriminator_logit(
                    discriminator,
                    Self::fake(generator, z),
                ))
            })
            .sum::<f64>()
            / NOISE.len() as f64
    }

    fn train(mut self, steps: usize) -> Self {
        for _ in 0..steps {
            let d_gradient = finite_gradient(&self.discriminator, |d| {
                Self::discriminator_loss(d, &self.generator)
            });
            update(&mut self.discriminator, d_gradient, 0.02);
            let g_gradient = finite_gradient(&self.generator, |g| {
                Self::generator_loss(g, &self.discriminator)
            });
            update(&mut self.generator, g_gradient, 0.02);
        }
        self
    }
}

#[derive(Clone, Copy, Debug)]
struct Diffusion {
    predictor: [f64; 3],
}

const ALPHA_BARS: [f64; 4] = [0.95, 0.6, 0.15, 0.02];

impl Diffusion {
    fn loss(self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for (time, alpha_bar) in ALPHA_BARS.into_iter().enumerate() {
            for &x0 in &DATA {
                for epsilon in NOISE {
                    let xt = alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * epsilon;
                    let time_feature = time as f64 / (ALPHA_BARS.len() - 1) as f64;
                    let predicted = self.predictor[0] * xt
                        + self.predictor[1] * time_feature
                        + self.predictor[2];
                    sum += (predicted - epsilon).powi(2);
                    count += 1;
                }
            }
        }
        sum / count as f64
    }

    fn train(mut self, steps: usize) -> Self {
        for _ in 0..steps {
            let gradient = finite_gradient(&self.predictor, |p| Self { predictor: *p }.loss());
            update(&mut self.predictor, gradient, 0.03);
        }
        self
    }

    fn sample(self, initial_noise: f64) -> f64 {
        let mut xt = initial_noise;
        for time in (0..ALPHA_BARS.len()).rev() {
            let time_feature = time as f64 / (ALPHA_BARS.len() - 1) as f64;
            let epsilon =
                self.predictor[0] * xt + self.predictor[1] * time_feature + self.predictor[2];
            xt = ddim_step(xt, time, epsilon);
        }
        xt
    }
}

fn ddim_step(xt: f64, time: usize, epsilon: f64) -> f64 {
    let alpha_bar = ALPHA_BARS[time];
    let x0_estimate = (xt - (1.0 - alpha_bar).sqrt() * epsilon) / alpha_bar.sqrt();
    let previous_alpha_bar = if time == 0 { 1.0 } else { ALPHA_BARS[time - 1] };
    previous_alpha_bar.sqrt() * x0_estimate + (1.0 - previous_alpha_bar).sqrt() * epsilon
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

fn main() {
    let vae_initial = Vae {
        p: [0.4, 0.0, -0.5, 0.4, 0.0],
    };
    let vae = vae_initial.train(500);
    let gan_initial = Gan {
        generator: [0.25, 0.3],
        discriminator: [0.2, 0.15, 0.0],
    };
    let gan = gan_initial.train(1_500);
    let diffusion_initial = Diffusion {
        predictor: [0.0; 3],
    };
    let diffusion = diffusion_initial.train(500);
    let mut rng = Rng(51);
    println!(
        "beta-VAE objective: {:.4} -> {:.4}; decoder means: {:.3}, {:.3}",
        Vae::loss_with(&vae_initial.p),
        Vae::loss_with(&vae.p),
        vae.decode(rng.normal()),
        vae.decode(rng.normal())
    );
    println!(
        "GAN generator: x={:.3}z{:+.3}; discriminator loss {:.4}",
        gan.generator[0],
        gan.generator[1],
        Gan::discriminator_loss(&gan.discriminator, &gan.generator)
    );
    println!(
        "diffusion noise MSE: {:.4} -> {:.4}; reverse sample {:.3}",
        diffusion_initial.loss(),
        diffusion.loss(),
        diffusion.sample(rng.normal())
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
    let diffusion_samples: Vec<_> = noise.iter().map(|&z| diffusion.sample(z)).collect();
    report_samples("data fixture", &DATA);
    report_samples("VAE decoder means", &vae_means);
    report_samples("VAE observations (unit decoder variance)", &vae_samples);
    report_samples("GAN", &gan_samples);
    report_samples("diffusion", &diffusion_samples);
    println!("All three models were trained here; affine Gaussian sampling cannot represent these two modes.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vae_and_diffusion_objectives_fall() {
        let vae = Vae {
            p: [0.4, 0.0, -0.5, 0.4, 0.0],
        };
        let trained_vae = vae.train(500);
        assert!(Vae::loss_with(&trained_vae.p) < Vae::loss_with(&vae.p) * 0.7);
        let diffusion = Diffusion {
            predictor: [0.0; 3],
        };
        assert!(diffusion.train(500).loss() < diffusion.loss() * 0.8);
    }

    #[test]
    fn gan_updates_both_networks_and_outputs_finite_samples() {
        let initial = Gan {
            generator: [0.25, 0.3],
            discriminator: [0.2, 0.15, 0.0],
        };
        let trained = initial.train(200);
        assert_ne!(trained.generator, initial.generator);
        assert_ne!(trained.discriminator, initial.discriminator);
        assert!(NOISE
            .iter()
            .all(|&z| Gan::fake(&trained.generator, z).is_finite()));
    }

    #[test]
    fn finite_difference_matches_quadratic_and_vae_reparameterization() {
        let gradient = finite_gradient(&[0.3, -0.7], |p| p[0].powi(2) + 3.0 * p[1].powi(2));
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
        assert!((Vae::loss_with(&p) - expected).abs() < 1e-12);
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
            assert!((ddim_step(xt, time, epsilon) - expected).abs() < 1e-12);
        }
    }
}
