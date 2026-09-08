#[cfg(test)]
fn normalize_log_terms(log_terms: &[f64]) -> Vec<f64> {
    let _ = log_terms;
    // TODO: subtract the maximum before exponentiating, then divide by the sum.
    todo!("compute stable responsibilities")
}

fn main() {
    let log_terms = [-1001.0_f64, -1000.0];
    let largest = log_terms.into_iter().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "largest component log term: {largest}; now implement stable normalization for the test"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn very_negative_log_terms_still_normalize() {
        let responsibilities = normalize_log_terms(&[-1001.0, -1000.0]);
        assert!((responsibilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(
            responsibilities[1] > responsibilities[0]
                && responsibilities.iter().all(|value| value.is_finite())
        );
    }
}
