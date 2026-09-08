//! Magnitude pruning, CSR conversion, scalar parity, and release-mode timing.
use std::hint::black_box;
use std::time::{Duration, Instant};

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

    fn density(&self) -> f64 {
        self.values.len() as f64 / (self.out_features * self.in_features) as f64
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

fn close(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(a, b)| {
            if !a.is_finite() || !b.is_finite() {
                return false;
            }
            let tolerance = 1e-10 + 1e-8 * a.abs().max(b.abs());
            (a - b).abs() <= tolerance
        })
}

fn median_range(mut samples: Vec<Duration>) -> (Duration, Duration, Duration) {
    samples.sort_unstable();
    (
        samples[samples.len() / 2],
        samples[0],
        samples[samples.len() - 1],
    )
}

fn dense_matvec_into(weights: &[f64], in_features: usize, input: &[f64], output: &mut [f64]) {
    for (value, row) in output.iter_mut().zip(weights.chunks_exact(in_features)) {
        *value = row.iter().zip(input).map(|(weight, x)| weight * x).sum();
    }
}

fn benchmark(
    pruned_weights: &[f64],
    csr: &CsrMatrix,
    input: &[f64],
    repeats: usize,
) -> (Duration, Duration, Duration, Duration, Duration, Duration) {
    let mut dense_output = vec![0.0; csr.out_features];
    let mut sparse_output = vec![0.0; csr.out_features];
    dense_matvec_into(pruned_weights, csr.in_features, input, &mut dense_output);
    csr.matvec_into(input, &mut sparse_output)
        .expect("validated fixture");
    black_box(&dense_output);
    black_box(&sparse_output);
    let mut dense_samples = Vec::with_capacity(repeats);
    let mut sparse_samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        dense_matvec_into(pruned_weights, csr.in_features, input, &mut dense_output);
        black_box(&dense_output);
        dense_samples.push(start.elapsed());
        let start = Instant::now();
        csr.matvec_into(input, &mut sparse_output)
            .expect("validated fixture");
        black_box(&sparse_output);
        sparse_samples.push(start.elapsed());
    }
    let dense_stats = median_range(dense_samples);
    let sparse_stats = median_range(sparse_samples);
    (
        dense_stats.0,
        dense_stats.1,
        dense_stats.2,
        sparse_stats.0,
        sparse_stats.1,
        sparse_stats.2,
    )
}

fn fixture(out_features: usize, in_features: usize) -> Vec<f64> {
    (0..out_features * in_features)
        .map(|i| (((i * 73 + 19) % 211) as f64 - 105.0) / 105.0)
        .collect()
}

fn root_mean_square_error(left: &[f64], right: &[f64]) -> f64 {
    (left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / left.len() as f64)
        .sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (out_features, in_features) = (192, 192);
    let weights = fixture(out_features, in_features);
    let input: Vec<_> = (0..in_features).map(|i| (i as f64 * 0.03).sin()).collect();
    let unpruned_output = dense_matvec_reference(&weights, out_features, in_features, &input)?;
    println!(
        "scalar CPU, f64, {out_features}x{in_features}, release={}",
        !cfg!(debug_assertions)
    );
    for requested in [1.0, 0.5, 0.2, 0.05] {
        let pruned = prune_to_density(&weights, requested)?;
        let csr = CsrMatrix::from_dense(&pruned, out_features, in_features)?;
        let dense_output = dense_matvec_reference(&pruned, out_features, in_features, &input)?;
        let sparse_output = csr.matvec(&input)?;
        if !close(&dense_output, &sparse_output) {
            return Err("dense and sparse kernels disagree".into());
        }
        let stats = benchmark(&pruned, &csr, &input, 31);
        let pruning_error = root_mean_square_error(&unpruned_output, &sparse_output);
        println!("density={:.3}, nnz={}, output RMSE={pruning_error:.5}, dense-kernel median/range={:?}/{:?}..{:?}, CSR-kernel median/range={:?}/{:?}..{:?}", csr.density(), csr.values.len(), stats.0, stats.1, stats.2, stats.3, stats.4, stats.5);
    }
    println!("These timings describe this scalar fixture and machine only.");
    Ok(())
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
