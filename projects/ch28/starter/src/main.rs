fn dot_scalar(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err("length mismatch".into());
    }
    Ok(a.iter().zip(b).map(|(x, y)| x * y).sum())
}
#[cfg(test)]
fn dot_dispatched(a: &[f32], b: &[f32]) -> Result<(&'static str, f32), String> {
    // TODO: keep this scalar fallback, then add runtime-guarded architecture branches from the lesson.
    let _ = (a, b);
    todo!("implement portable dispatch")
}
fn main() {
    let a = [1., 2., 3.];
    let b = [4., -1., 0.5];
    println!("prior scalar dot={}", dot_scalar(&a, &b).unwrap());
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dispatch_preserves_scalar_answer() {
        let a = [1., 2., 3., 4., 5.];
        let b = [-1., 0.5, 2., 0.25, -0.5];
        let (_, got) = dot_dispatched(&a, &b).unwrap();
        assert!((got - dot_scalar(&a, &b).unwrap()).abs() < 1e-6);
    }
}
