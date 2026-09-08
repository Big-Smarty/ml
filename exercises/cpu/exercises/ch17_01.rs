fn projection_score(centered: [f64; 2], component: [f64; 2]) -> f64 {
    // TODO: return the PCA dot product q^T (x - mean). Pca1::transform performs the
    // centering first; this arithmetic primitive receives x - mean directly.
    let _ = (centered, component);
    todo!("project onto the component")
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
