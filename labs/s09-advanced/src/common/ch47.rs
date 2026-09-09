// Magnitude pruning, CSR conversion, scalar parity, and release-mode timing.
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

fn demo() -> Result<(), Box<dyn std::error::Error>> {
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
