//! Worked solution: Rank original indices by descending magnitude with deterministic ties; retain coordinates rather than sorting values. CSR omits exact zeros and appends a pointer even for an empty row. Its kernel dereferences original column indices, and the in-place variant avoids timed output allocation.
//! Compare intermediate quantities with the lesson before using the final goal check.
//! Worked solution for chapter 47. Read the comments and lesson explanations before comparing.
include!("../common/ch47.rs");
include!("../checks/ch47.rs");
fn prune_to_density(weights: &[f64], density: f64) -> Result<Vec<f64>, &'static str> {
    if weights.is_empty()
        || weights.iter().any(|value| !value.is_finite())
        || !density.is_finite()
        || !(0.0..=1.0).contains(&density)
    {
        return Err("pruning needs finite weights and density in [0, 1]");
    }
    let keep = (density * weights.len() as f64).round() as usize;
    let mut order: Vec<_> = (0..weights.len()).collect();
    order.sort_by(|&left, &right| {
        weights[right]
            .abs()
            .total_cmp(&weights[left].abs())
            .then_with(|| left.cmp(&right))
    });
    let mut mask = vec![false; weights.len()];
    order[..keep].iter().for_each(|&index| mask[index] = true);
    Ok(weights
        .iter()
        .zip(mask)
        .map(|(&weight, keep)| if keep { weight } else { 0.0 })
        .collect())
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
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_matches_masked_dense_including_empty_rows() -> Result<(), &'static str> {
        let dense = vec![0.0, 2.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 4.0];
        let csr = CsrMatrix::from_dense(&dense, 3, 3)?;
        assert_eq!(csr.row_ptr, [0, 1, 1, 3]);
        assert_eq!(csr.col_idx, [1, 0, 2]);
        assert_eq!(csr.values, [2.0, -3.0, 4.0]);
        assert!(close(
            &dense_matvec_reference(&dense, 3, 3, &[1.0, 2.0, 3.0])?,
            &csr.matvec(&[1.0, 2.0, 3.0])?
        ));
        let mut dense_output = [0.0; 3];
        let mut sparse_output = [0.0; 3];
        dense_matvec_into(&dense, 3, &[1.0, 2.0, 3.0], &mut dense_output);
        csr.matvec_into(&[1.0, 2.0, 3.0], &mut sparse_output)?;
        assert!(close(&dense_output, &[4.0, 0.0, 9.0]));
        assert!(close(&sparse_output, &dense_output));
        Ok(())
    }

    #[test]
    fn pruning_keeps_requested_largest_magnitudes() -> Result<(), &'static str> {
        let pruned = prune_to_density(&[-1.0, 0.2, 3.0, -2.0], 0.5)?;
        assert_eq!(pruned, [0.0, 0.0, 3.0, -2.0]);
        assert_eq!(prune_to_density(&[2.0, -2.0], 0.5)?, [2.0, 0.0]);
        Ok(())
    }

    #[test]
    fn invalid_shapes_are_rejected() {
        assert!(CsrMatrix::from_dense(&[1.0], 1, 2).is_err());
        assert!(dense_matvec_reference(&[1.0], 1, 1, &[]).is_err());
    }
}
