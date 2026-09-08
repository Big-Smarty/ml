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

fn shapes(a: &[f32], b: &[f32], c: &[f32], m: usize, k: usize, n: usize) -> Result<(), String> {
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

fn matmul_ijk(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    shapes(a, b, c, m, k, n)?;
    c.fill(0.0);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    Ok(())
}

fn matmul_ikj(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    shapes(a, b, c, m, k, n)?;
    c.fill(0.0);
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j];
            }
        }
    }
    Ok(())
}

fn matmul_blocked(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    block: usize,
) -> Result<(), String> {
    shapes(a, b, c, m, k, n)?;
    if block == 0 {
        return Err("block size must be positive".into());
    }
    c.fill(0.0);
    for ii in (0..m).step_by(block) {
        for pp in (0..k).step_by(block) {
            for jj in (0..n).step_by(block) {
                for i in ii..ii.saturating_add(block).min(m) {
                    for p in pp..pp.saturating_add(block).min(k) {
                        let av = a[i * k + p];
                        for j in jj..jj.saturating_add(block).min(n) {
                            c[i * n + j] += av * b[p * n + j];
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
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                c[i * n + j] += a[i * k + p] as f64 * b[p * n + j] as f64;
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
            matmul_ijk as fn(&[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<(), String>,
        ),
        ("ikj", matmul_ikj),
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
        || matmul_blocked(black_box(&a), black_box(&b), black_box(&mut c), m, k, n, 32).unwrap(),
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
            matmul_ijk as fn(&[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<(), String>,
            matmul_ikj,
        ] {
            let mut c = vec![9.; m * n];
            f(&a, &b, &mut c, m, k, n).unwrap();
            assert!(close(&c, &r, k));
        }
        for block in [1, 4, 16, usize::MAX] {
            let mut c = vec![9.; m * n];
            matmul_blocked(&a, &b, &mut c, m, k, n, block).unwrap();
            assert!(close(&c, &r, k));
        }
    }
    #[test]
    fn rejects_bad_shapes_and_zero_block() {
        assert!(matmul_ijk(&[1.], &[1.], &mut [0.], 2, 1, 1).is_err());
        assert!(matmul_blocked(&[1.], &[1.], &mut [0.], 1, 1, 1, 0).is_err());
    }
}
