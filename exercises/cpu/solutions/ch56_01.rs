fn capacity(tokens: usize, experts: usize, factor: f64) -> usize {
    assert!(experts > 0 && factor.is_finite() && factor > 0.0);
    (((tokens as f64 / experts as f64) * factor).ceil() as usize).max(1)
}

fn main() {
    println!("capacity: {}", capacity(7, 3, 1.25));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_up_and_keeps_one_slot() {
        assert_eq!(capacity(7, 3, 1.25), 3);
        assert_eq!(capacity(1, 8, 0.5), 1);
    }
}
