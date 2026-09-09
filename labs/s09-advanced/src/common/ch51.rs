// Three tiny trainable generative models with deterministic teaching fixtures.
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

#[derive(Clone, Copy, Debug)]
struct Vae {
    parameters: [f64; 5],
}

impl Vae {
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

fn demo() -> Result<(), &'static str> {
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
        "selected autoencoder objective: {:.4} -> {:.4}; decoder means: {:.3}, {:.3}",
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
    report_samples("latent decoder means", &vae_means);
    report_samples(
        "latent decoder observations (unit decoder variance)",
        &vae_samples,
    );
    report_samples("GAN", &gan_samples);
    report_samples("diffusion", &diffusion_samples);
    println!("All three models were trained here; affine Gaussian sampling cannot represent these two modes.");
    Ok(())
}
