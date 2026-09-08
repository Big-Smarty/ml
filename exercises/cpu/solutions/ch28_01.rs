fn scalar_tail(a: &[f32], b: &[f32], consumed: usize) -> f32 {
    a[consumed..]
        .iter()
        .zip(&b[consumed..])
        .map(|(x, y)| x * y)
        .sum()
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
