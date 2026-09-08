fn projection_score(centered: [f64; 2], component: [f64; 2]) -> f64 {
    centered[0] * component[0] + centered[1] * component[1]
}

fn main() {
    println!(
        "{}",
        projection_score([2.0, 0.0], [0.5_f64.sqrt(), 0.5_f64.sqrt()])
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection_matches_hand_calculation() {
        let q = [1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];
        assert!((projection_score([2.0, 0.0], q) - 2.0_f64.sqrt()).abs() < 1e-12);
    }
}
