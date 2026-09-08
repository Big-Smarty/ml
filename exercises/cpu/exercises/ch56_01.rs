fn capacity(tokens: usize, experts: usize, factor: f64) -> usize {
    // TODO: implement ceil(tokens / experts * factor), with a minimum of one slot.
    todo!("convert after division, then call ceil")
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
        assert_eq!(capacity(8, 3, 1.25), 4);
        assert_eq!(capacity(1, 8, 0.5), 1);
    }
}
