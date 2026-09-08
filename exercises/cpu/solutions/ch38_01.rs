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
        let scale = (f64::from(limit) / norm) as f32;
        for value in gradient {
            *value *= scale;
        }
    }
    Ok(())
}
fn main() {
    println!("Clip one global gradient after accumulation.");
}
#[test]
fn clips_norm() {
    let mut g = [3.0, 4.0];
    clip_global_norm(&mut g, 1.0).unwrap();
    assert!((g[0] - 0.6).abs() < 1e-6 && (g[1] - 0.8).abs() < 1e-6);
    assert!(clip_global_norm(&mut [f32::NAN], 1.0).is_err());
}
