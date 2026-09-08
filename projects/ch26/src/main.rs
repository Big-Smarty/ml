//! Three allocation-free row-major matrix multiplication loop organizations.
use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

fn values(len: usize, salt: u32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as u32 * 97 + salt) % 101) as f32 - 50.0) / 50.0)
        .collect()
}

fn validate_matmul_shapes(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    if m == 0
        || k == 0
        || n == 0
        || a.len() != m.checked_mul(k).ok_or("shape overflow")?
        || b.len() != k.checked_mul(n).ok_or("shape overflow")?
        || c.len() != m.checked_mul(n).ok_or("shape overflow")?
    {
        Err("expected A=[m,k], B=[k,n], C=[m,n] with positive dimensions".into())
    } else {
        Ok(())
    }
}

fn matmul_scalar_row_col_inner(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    validate_matmul_shapes(a, b, c, m, k, n)?;
    c.fill(0.0);
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    Ok(())
}

fn matmul_scalar_row_inner_col(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    validate_matmul_shapes(a, b, c, m, k, n)?;
    c.fill(0.0);
    for row in 0..m {
        for inner in 0..k {
            let a_value = a[row * k + inner];
            for col in 0..n {
                c[row * n + col] += a_value * b[inner * n + col];
            }
        }
    }
    Ok(())
}

fn matmul_blocked_row_inner_col(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    block: usize,
) -> Result<(), String> {
    // ponytail: one scalar tile level; add packing/register kernels only after measurements justify them.
    validate_matmul_shapes(a, b, c, m, k, n)?;
    if block == 0 {
        return Err("block size must be positive".into());
    }
    c.fill(0.0);
    for row_start in (0..m).step_by(block) {
        for inner_start in (0..k).step_by(block) {
            for col_start in (0..n).step_by(block) {
                for row in row_start..row_start.saturating_add(block).min(m) {
                    for inner in inner_start..inner_start.saturating_add(block).min(k) {
                        let a_value = a[row * k + inner];
                        for col in col_start..col_start.saturating_add(block).min(n) {
                            c[row * n + col] += a_value * b[inner * n + col];
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn reference_f64(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            for inner in 0..k {
                c[row * n + col] += a[row * k + inner] as f64 * b[inner * n + col] as f64;
            }
        }
    }
    c
}

fn close(got: &[f32], want: &[f64], k: usize) -> bool {
    let atol = 2e-5 * k as f64;
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(&a, &b)| (a as f64 - b).abs() <= atol + 2e-5 * b.abs())
}

fn time_kernel(mut run: impl FnMut(), repetitions: usize) -> (Duration, Duration, Duration) {
    for _ in 0..3 {
        run();
    }
    let mut times = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let t = Instant::now();
        run();
        times.push(t.elapsed());
    }
    times.sort_unstable();
    (times[0], times[times.len() / 2], times[times.len() - 1])
}

fn experiment(m: usize, k: usize, n: usize, repetitions: usize) -> Result<(), String> {
    println!("GEMM m={m} k={k} n={n}; f32; threads=1; architecture={} OS={}; warmups=3 samples={repetitions}; timing includes output clearing and shape checks", env::consts::ARCH, env::consts::OS);
    let a = values(m * k, 7);
    let b = values(k * n, 29);
    let reference = reference_f64(&a, &b, m, k, n);
    let mut c = vec![0.0; m * n];
    for (name, kernel) in [
        (
            "ijk",
            matmul_scalar_row_col_inner
                as fn(&[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<(), String>,
        ),
        ("ikj", matmul_scalar_row_inner_col),
    ] {
        let (lo, med, hi) = time_kernel(
            || kernel(black_box(&a), black_box(&b), black_box(&mut c), m, k, n).unwrap(),
            repetitions,
        );
        if !close(&c, &reference, k) {
            return Err(format!("{name} disagrees with f64 reference"));
        }
        println!(
            "{name:>7}: median={med:?} range={lo:?}..{hi:?} GFLOP/s={:.3}",
            2.0 * m as f64 * k as f64 * n as f64 / med.as_secs_f64() / 1e9
        );
    }
    let (lo, med, hi) = time_kernel(
        || {
            matmul_blocked_row_inner_col(
                black_box(&a),
                black_box(&b),
                black_box(&mut c),
                m,
                k,
                n,
                32,
            )
            .unwrap()
        },
        repetitions,
    );
    if !close(&c, &reference, k) {
        return Err("blocked disagrees with f64 reference".into());
    }
    println!(
        "blocked: median={med:?} range={lo:?}..{hi:?} GFLOP/s={:.3}; block=32; checksum={:.6}",
        2.0 * m as f64 * k as f64 * n as f64 / med.as_secs_f64() / 1e9,
        black_box(&c).iter().map(|&v| v as f64).sum::<f64>()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = env::args().collect();
    match a.as_slice() {
        [_] => experiment(37, 29, 31, 9)?,
        [_, f] if f == "--large" => experiment(257, 259, 263, 15)?,
        _ => return Err("usage: ch26 [--large]".into()),
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_orders_handle_rectangular_odd_shapes() {
        let (m, k, n) = (3, 5, 7);
        let a = values(m * k, 7);
        let b = values(k * n, 29);
        let r = reference_f64(&a, &b, m, k, n);
        for f in [
            matmul_scalar_row_col_inner
                as fn(&[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<(), String>,
            matmul_scalar_row_inner_col,
        ] {
            let mut c = vec![9.; m * n];
            f(&a, &b, &mut c, m, k, n).unwrap();
            assert!(close(&c, &r, k));
        }
        for block in [1, 4, 16, usize::MAX] {
            let mut c = vec![9.; m * n];
            matmul_blocked_row_inner_col(&a, &b, &mut c, m, k, n, block).unwrap();
            assert!(close(&c, &r, k));
        }
    }
    #[test]
    fn rejects_bad_shapes_and_zero_block() {
        assert!(matmul_scalar_row_col_inner(&[1.], &[1.], &mut [0.], 2, 1, 1).is_err());
        assert!(matmul_blocked_row_inner_col(&[1.], &[1.], &mut [0.], 1, 1, 1, 0).is_err());
    }

    #[test]
    fn dense_weight_transpose_maps_to_gemm_b() {
        let (m, k, n) = (2, 3, 2);
        let inputs = [1., 2., 3., 4., 5., 6.];
        let weights = [1., 3., 5., 2., 4., 6.]; // [out_features,in_features]
        let mut b = [0.; 6]; // weights-transpose, [k,n]
        for inner in 0..k {
            for col in 0..n {
                b[inner * n + col] = weights[col * k + inner];
            }
        }
        let mut outputs = [0.; 4];
        matmul_scalar_row_inner_col(&inputs, &b, &mut outputs, m, k, n).unwrap();
        assert_eq!(outputs, [22., 28., 49., 64.]);
    }
}
