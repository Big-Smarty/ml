pub fn check() -> Result<(), String> {
    crate::ensure(
        crate::close(reparameterize(0.8, 0.25_f64.ln(), -0.5), 0.55),
        "reparameterization must exponentiate half the log variance",
    )?;
    let p = [0.4, 0.1, -0.5, 0.7, -0.2];
    let vae = Vae { parameters: p };
    let variance = NOISE.iter().map(|e| e * e).sum::<f64>() / NOISE.len() as f64;
    let expected = DATA
        .iter()
        .map(|&x| {
            let mean = p[0] * x + p[1];
            0.5 * ((p[3] * mean + p[4] - x).powi(2) + p[3].powi(2) * p[2].exp() * variance)
                + 0.05 * (mean.powi(2) + p[2].exp() - 1.0 - p[2])
        })
        .sum::<f64>()
        / DATA.len() as f64;
    let actual = vae.loss(&DATA, &NOISE)?;
    crate::ensure(
        crate::close(actual, expected),
        &format!("VAE objective {actual:.8} differs from reconstruction plus 0.1 KL {expected:.8}"),
    )?;
    crate::ensure(
        crate::close(
            Gan::generator_loss(&[0.0; 2], &[0.0; 3], &[-0.3, 0.8])?,
            2.0_f64.ln(),
        ),
        "non-saturating GAN generator loss at D=0.5 must be +ln2",
    )?;
    crate::ensure(
        crate::close(
            Gan::discriminator_loss(&[0.0; 3], &[0.0; 2], &[-1.0, 1.0], &[-0.3, 0.8])?,
            4.0_f64.ln(),
        ),
        "discriminator loss at D=0.5 must be ln4",
    )?;
    for (time, &a) in ALPHA_BARS.iter().enumerate() {
        let (x0, eps) = (0.7, -0.4);
        let xt = a.sqrt() * x0 + (1.0 - a).sqrt() * eps;
        let prev = if time == 0 { 1.0 } else { ALPHA_BARS[time - 1] };
        crate::ensure(
            crate::close(
                ddim_step(xt, time, eps, &ALPHA_BARS)?,
                prev.sqrt() * x0 + (1.0 - prev).sqrt() * eps,
            ),
            "perfect-noise DDIM transition failed",
        )?;
    }
    let model = Diffusion {
        predictor: [0.0; 3],
    };
    let trained = model.train(&DATA, &NOISE, &ALPHA_BARS, 500, 0.03)?;
    crate::ensure(
        trained.loss(&DATA, &NOISE, &ALPHA_BARS)? < model.loss(&DATA, &NOISE, &ALPHA_BARS)? * 0.8,
        "diffusion predictor did not learn injected noise",
    )
}

pub fn run(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("this experiment takes no extra arguments".into());
    }
    demo().map_err(|e| e.to_string())
}
