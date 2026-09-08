#[derive(Debug)]
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
        // TODO: validate the [out_features, in_features] shape, then scan each
        // output row and record its nonzero values, columns, and ending offset.
        let _ = (weights, out_features, in_features);
        todo!()
    }
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_preserves_coordinates_and_empty_rows() -> Result<(), &'static str> {
        let weights = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 4.0];
        let csr = CsrMatrix::from_dense(&weights, 3, 3)?;
        assert_eq!((csr.out_features, csr.in_features), (3, 3));
        assert_eq!(csr.row_ptr, [0, 1, 1, 3]);
        assert_eq!(csr.col_idx, [1, 0, 2]);
        assert_eq!(csr.values, [2.0, -3.0, 4.0]);
        Ok(())
    }
}
