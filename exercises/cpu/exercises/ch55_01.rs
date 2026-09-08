fn close(left: &[f32], right: &[f32], atol: f32, rtol: f32) -> bool {
    // TODO: implement the absolute-plus-relative parity rule.
    todo!("compare each aligned pair")
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
