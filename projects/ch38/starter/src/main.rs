use ch36::{Config, Decoder};
fn clip_global_norm(gradient: &mut [f32], limit: f32) -> Result<(), &'static str> {
    if !limit.is_finite() || limit <= 0.0 || gradient.iter().any(|value| !value.is_finite()) {
        return Err("gradient values or clipping limit are invalid");
    }
    let norm = gradient
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() {
        return Err("nonfinite gradient norm");
    }
    if norm > f64::from(limit) {
        // TODO: compute the f64 limit/norm scale, cast it to f32, and scale every element.
        todo!("guided repair: scale the complete gradient")
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _exercise = clip_global_norm as fn(&mut [f32], f32) -> Result<(), &'static str>;
    let model = Decoder::new(
        Config {
            vocab_size: 256,
            context: 8,
            width: 16,
            heads: 2,
            layers: 1,
            ff_width: 32,
        },
        38,
    )?;
    println!(
        "Decoder has {} parameters; complete global clipping, then test.",
        model.parameter_count()
    );
    Ok(())
}
#[test]
fn clips_global_norm() {
    let mut g = [3.0, 4.0];
    clip_global_norm(&mut g, 1.0).unwrap();
    assert!((g[0] - 0.6).abs() < 1e-6 && (g[1] - 0.8).abs() < 1e-6);
    assert!(clip_global_norm(&mut [f32::NAN], 1.0).is_err());
}
