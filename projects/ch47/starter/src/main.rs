//! Keep the dense oracle working while implementing sparse conversion.

fn dense_matvec(matrix: &[f64], cols: usize, input: &[f64]) -> Vec<f64> {
    matrix
        .chunks_exact(cols)
        .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum())
        .collect()
}

fn dense_to_csr_values(dense: &[f64]) -> Vec<f64> {
    // TODO: preserve row-major order while omitting zeros.
    let _ = dense;
    todo!("collect only nonzero values in row-major order")
}

fn main() {
    let _guided_todo: fn(&[f64]) -> Vec<f64> = dense_to_csr_values;
    let matrix = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 4.0];
    println!(
        "dense oracle output: {:?}",
        dense_matvec(&matrix, 3, &[1.0, 2.0, 3.0])
    );
    println!("Now run cargo test and implement dense_to_csr_values.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_values_skip_structural_zeros() {
        assert_eq!(dense_to_csr_values(&[0.0, 2.0, 0.0, -3.0]), [2.0, -3.0]);
    }
}
