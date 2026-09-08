fn dense_matvec_reference(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    input: &[f32],
) -> Vec<f32> {
    (0..out_features)
        .map(|r| {
            (0..in_features)
                .map(|c| weights[r * in_features + c] * input[c])
                .sum()
        })
        .collect()
}
fn int8_scale(_row: &[f32]) -> f32 {
    // TODO: return max(abs(row)) / 127, using 1 for an all-zero row.
    todo!("compute a symmetric per-row scale")
}
fn main() {
    let _guided: fn(&[f32]) -> f32 = int8_scale;
    let weights = [1.0, -2.0, 0.5, 3.0];
    println!(
        "checkpoint dense output: {:?}",
        dense_matvec_reference(&weights, 2, 2, &[2.0, -1.0])
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scale_uses_largest_magnitude() {
        assert!((int8_scale(&[-2.54, 1.0]) - 0.02).abs() < 1e-6);
    }
}
