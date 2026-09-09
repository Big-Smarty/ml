//! Chapter 47 learner algorithms. Baseline keeps every weight and stores explicit zeros. Build global magnitude pruning, zero-omitting CSR construction, and sparse matvec; the representation check and pruning check are distinct.
include!("common/ch47.rs");
include!("checks/ch47.rs");
fn prune_to_density(weights: &[f64], density: f64) -> Result<Vec<f64>, &'static str> {
    if weights.is_empty()
        || weights.iter().any(|value| !value.is_finite())
        || !density.is_finite()
        || !(0.0..=1.0).contains(&density)
    {
        return Err("pruning needs finite weights and density in [0, 1]");
    }
    Ok(weights.to_vec())
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
        let mut row_ptr = Vec::with_capacity(out_features + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for row in weights.chunks_exact(in_features) {
            for (column, &value) in row.iter().enumerate() {
                {
                    // Baseline: store every position, including exact zeros.
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

impl CsrMatrix {
    fn matvec(&self, input: &[f64]) -> Result<Vec<f64>, &'static str> {
        if input.len() != self.in_features || input.iter().any(|value| !value.is_finite()) {
            return Err("input length must match columns and values must be finite");
        }
        Ok((0..self.out_features)
            .map(|row| {
                (self.row_ptr[row]..self.row_ptr[row + 1])
                    .map(|i| self.values[i] * input[self.col_idx[i]])
                    .sum()
            })
            .collect())
    }
}

impl CsrMatrix {
    fn matvec_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), &'static str> {
        if input.len() != self.in_features || output.len() != self.out_features {
            return Err("matvec buffers do not match the matrix shape");
        }
        for (row, value) in output.iter_mut().enumerate() {
            *value = (self.row_ptr[row]..self.row_ptr[row + 1])
                .map(|i| self.values[i] * input[self.col_idx[i]])
                .sum();
        }
        Ok(())
    }
}
