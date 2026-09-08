fn close(left: &[f32], right: &[f32], atol: f32, rtol: f32) -> bool {
    atol.is_finite()
        && atol >= 0.0
        && rtol.is_finite()
        && rtol >= 0.0
        && left.len() == right.len()
        && left.iter().zip(right).all(|(&a, &b)| {
            let scale = a.abs().max(b.abs());
            a.is_finite() && b.is_finite() && (a - b).abs() <= atol + rtol * scale
        })
}

fn main() {
    println!(
        "parity: {}",
        close(&[1.0, 2.0], &[1.0, 2.00001], 1e-5, 1e-5)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_lengths_and_tolerances() {
        assert!(close(&[1.0, 1000.0], &[1.000001, 1000.009], 1e-5, 1e-5));
        assert!(!close(&[1.0], &[1.0, 2.0], 1e-5, 1e-5));
        assert!(!close(&[1.0], &[1.1], 1e-5, 1e-5));
        assert!(!close(&[f32::INFINITY], &[0.0], 1e-5, 1e-5));
        assert!(!close(&[f32::NAN], &[0.0], 1e-5, 1e-5));
        assert!(!close(&[1.0], &[1.0], -1.0, 1.0));
    }
}
