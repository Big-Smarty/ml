fn median(_samples: &mut [u128]) -> u128 {
    // TODO: report a typical repeated measurement, not the single fastest sample.
    todo!("sort and return the middle sample")
}

fn main() {
    let mut samples = [15, 9, 11, 10, 40];
    println!("median: {} ns", median(&mut samples));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignores_one_fast_and_one_slow_outlier() {
        let mut samples = [15, 9, 11, 10, 40];
        assert_eq!(median(&mut samples), 11);
    }
}
