fn autoencode(x: f64, encoder: f64, decoder: f64) -> f64 {
    decoder * (encoder * x).tanh()
}

#[cfg(test)]
fn reparameterize(mean: f64, log_variance: f64, epsilon: f64) -> f64 {
    // TODO: return mean + standard_deviation * epsilon.
    let _ = (mean, log_variance, epsilon);
    todo!("convert log variance to standard deviation")
}

fn main() {
    for x in [-1.0, 0.0, 1.0] {
        println!("x={x:+.1} reconstruction={:+.3}", autoencode(x, 0.8, 1.2));
    }
    println!("This deterministic autoencoder is the checkpoint before a stochastic VAE.");
}

#[test]
fn zero_log_variance_has_unit_scale() {
    assert!((reparameterize(2.0, 0.0, -0.5) - 1.5).abs() < 1e-12);
}

#[test]
fn quarter_variance_halves_noise_scale() {
    assert!((reparameterize(0.8, 0.25_f64.ln(), -0.5) - 0.55).abs() < 1e-12);
}
