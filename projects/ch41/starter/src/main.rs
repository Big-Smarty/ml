fn dense(weights: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|r| (0..cols).map(|c| weights[r * cols + c] * x[c]).sum())
        .collect()
}
fn int8_scale(_row: &[f32]) -> f32 {
    // TODO: return max(abs(row)) / 127, using 1 for an all-zero row.
    todo!("compute a symmetric per-row scale")
}
fn main() {
    let _guided: fn(&[f32]) -> f32 = int8_scale;
    let w = [1.0, -2.0, 0.5, 3.0];
    println!(
        "checkpoint dense output: {:?}",
        dense(&w, 2, 2, &[2.0, -1.0])
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
