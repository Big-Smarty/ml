fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
#[cfg(test)]
fn dot_dispatch(a: &[f32], b: &[f32]) -> Result<(&'static str, f32), String> {
    if a.len() != b.len() {
        return Err("dot-product vectors must have equal lengths".into());
    }
    // TODO: keep this scalar fallback, then add runtime-guarded architecture branches from the lesson.
    let _ = (a, b);
    todo!("implement portable dispatch")
}
fn main() {
    let a = [1., 2., 3.];
    let b = [4., -1., 0.5];
    println!("prior scalar dot={}", dot_scalar(&a, &b));
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dispatch_preserves_scalar_answer() {
        let a = [1., 2., 3., 4., 5.];
        let b = [-1., 0.5, 2., 0.25, -0.5];
        let (_, got) = dot_dispatch(&a, &b).unwrap();
        assert!((got - dot_scalar(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn dispatch_rejects_unequal_lengths() {
        assert!(dot_dispatch(&[1.0], &[1.0, 2.0]).is_err());
    }
}
