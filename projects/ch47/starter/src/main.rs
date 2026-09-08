//! Keep the dense oracle working while implementing sparse conversion.

#[derive(Debug)]
#[allow(
    dead_code,
    reason = "fields are read after the guided CSR conversion TODO is implemented"
)]
struct CsrMatrix {
    out_features: usize,
    in_features: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<f64>,
}

impl CsrMatrix {
    fn from_dense(
        weights: &[f64],
        out_features: usize,
        in_features: usize,
    ) -> Result<Self, &'static str> {
        // TODO: validate the [out_features, in_features] shape, then collect
        // values, column indices, and one cumulative row offset per output row.
        let _ = (weights, out_features, in_features);
        todo!("convert row-major weights to CSR")
    }
}

fn dense_matvec_reference(
    weights: &[f64],
    out_features: usize,
    in_features: usize,
    input: &[f64],
) -> Result<Vec<f64>, &'static str> {
    if out_features == 0
        || in_features == 0
        || out_features.checked_mul(in_features) != Some(weights.len())
        || input.len() != in_features
        || weights.iter().chain(input).any(|value| !value.is_finite())
    {
        return Err("matrix and input shapes do not align");
    }
    Ok(weights
        .chunks_exact(in_features)
        .map(|row| row.iter().zip(input).map(|(weight, x)| weight * x).sum())
        .collect())
}

fn main() -> Result<(), &'static str> {
    let _guided_todo: fn(&[f64], usize, usize) -> Result<CsrMatrix, &'static str> =
        CsrMatrix::from_dense;
    let weights = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 4.0];
    println!(
        "dense oracle output: {:?}",
        dense_matvec_reference(&weights, 3, 3, &[1.0, 2.0, 3.0])?
    );
    println!("Now run cargo test and implement CsrMatrix::from_dense.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_preserves_csr_coordinates_and_empty_rows() -> Result<(), &'static str> {
        let weights = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 4.0];
        let csr = CsrMatrix::from_dense(&weights, 3, 3)?;
        assert_eq!((csr.out_features, csr.in_features), (3, 3));
        assert_eq!(csr.row_ptr, [0, 1, 1, 3]);
        assert_eq!(csr.col_idx, [1, 0, 2]);
        assert_eq!(csr.values, [2.0, -3.0, 4.0]);
        Ok(())
    }
}
