fn scalar_tail(a: &[f32], b: &[f32], consumed: usize) -> f32 {
    // TODO: multiply every element after the complete SIMD-width prefix.
    let _ = (a, b, consumed);
    todo!("finish the tail without reading past either slice")
}

fn main() {
    println!("tail={}", scalar_tail(&[1., 2., 3.], &[4., 5., 6.], 2));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_empty_and_nonempty_tails() {
        assert_eq!(scalar_tail(&[1., 2.], &[3., 4.], 2), 0.0);
        assert_eq!(scalar_tail(&[1., 2., 3.], &[4., 5., 6.], 2), 18.0);
    }
}
