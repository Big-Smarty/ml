use ch36::{Config, Decoder};
fn clip(grad: &mut [f32], limit: f32) {
    let norm = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > limit {
        // TODO: multiply every element by limit / norm.
        todo!("guided repair: scale the complete gradient")
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _exercise = clip as fn(&mut [f32], f32);
    let model = Decoder::new(Config::tiny(), 38)?;
    println!(
        "Decoder has {} parameters; complete global clipping, then test.",
        model.parameter_count()
    );
    Ok(())
}
#[test]
fn clips_global_norm() {
    let mut g = [3.0, 4.0];
    clip(&mut g, 1.0);
    assert!((g[0] - 0.6).abs() < 1e-6 && (g[1] - 0.8).abs() < 1e-6);
}
