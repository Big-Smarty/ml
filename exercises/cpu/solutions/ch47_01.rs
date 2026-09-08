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
        if out_features == 0
            || in_features == 0
            || out_features.checked_mul(in_features) != Some(weights.len())
        {
            return Err("dense shape must be nonzero and match its data");
        }
        if weights.iter().any(|value| !value.is_finite()) {
            return Err("matrix values must be finite");
        }
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for row in weights.chunks_exact(in_features) {
            for (column, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    col_idx.push(column);
                    values.push(value);
                }
            }
            row_ptr.push(values.len());
        }
        Ok(Self {
            out_features,
            in_features,
            row_ptr,
            col_idx,
            values,
        })
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
