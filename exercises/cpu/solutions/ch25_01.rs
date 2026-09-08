fn median_ns(samples: &mut [u128]) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    let mut samples = [15, 9, 11, 10, 40];
    println!("median: {} ns", median_ns(&mut samples));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignores_one_fast_and_one_slow_outlier() {
        let mut samples = [15, 9, 11, 10, 40];
        assert_eq!(median_ns(&mut samples), 11);
    }
}
