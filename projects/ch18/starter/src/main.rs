#[cfg(test)]
fn normalize_log_weights(log_weights: &[f64]) -> Vec<f64> {
    let _ = log_weights;
    // TODO: subtract the maximum before exponentiating, then divide by the sum.
    todo!("compute stable responsibilities")
}

fn main() {
    let log_weights = [-1001.0_f64, -1000.0];
    let largest = log_weights.into_iter().fold(f64::NEG_INFINITY, f64::max);
    println!("largest log weight: {largest}; now implement stable normalization for the test");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn very_negative_log_weights_still_normalize() {
        let r = normalize_log_weights(&[-1001.0, -1000.0]);
        assert!((r.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(r[1] > r[0] && r.iter().all(|x| x.is_finite()));
    }
}
