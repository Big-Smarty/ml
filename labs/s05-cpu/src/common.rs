use std::{hint::black_box, time::Duration};
pub type Run = fn(&[String]) -> Result<(), String>;
pub type Check = fn() -> Result<(), String>;
/// Shape of X[m,k], stored weights W[n,k], and output Y[m,n].
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}
pub fn values(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i.wrapping_mul(73).wrapping_add(salt)) % 257) as f32 - 128.) / 128.)
        .collect()
}
pub fn validate(x: &[f32], w: &[f32], bias: &[f32], y: &[f32], s: Shape) -> Result<(), String> {
    if s.m == 0
        || s.k == 0
        || s.n == 0
        || x.len() != s.m.checked_mul(s.k).ok_or("shape overflow")?
        || w.len() != s.n.checked_mul(s.k).ok_or("shape overflow")?
        || bias.len() != s.n
        || y.len() != s.m.checked_mul(s.n).ok_or("shape overflow")?
    {
        return Err("expected positive X[m,k], W[n,k], bias[n], Y[m,n]".into());
    }
    if x.iter().chain(w).chain(bias).any(|v| !v.is_finite()) {
        return Err("inputs and parameters must be finite".into());
    }
    Ok(())
}
pub fn finite(y: &[f32]) -> Result<(), String> {
    if y.iter().any(|v| !v.is_finite()) {
        Err("output overflow; rescale inputs".into())
    } else {
        Ok(())
    }
}
pub fn dense(x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32], s: Shape) -> Result<(), String> {
    validate(x, w, bias, y, s)?;
    for row in 0..s.m {
        for col in 0..s.n {
            let mut sum = bias[col];
            for inner in 0..s.k {
                sum += x[row * s.k + inner] * w[col * s.k + inner];
            }
            y[row * s.n + col] = sum;
        }
    }
    finite(y)
}
pub fn oracle(x: &[f32], w: &[f32], bias: &[f32], s: Shape) -> Vec<f64> {
    (0..s.m * s.n)
        .map(|index| {
            let (row, col) = (index / s.n, index % s.n);
            bias[col] as f64
                + (0..s.k)
                    .map(|i| x[row * s.k + i] as f64 * w[col * s.k + i] as f64)
                    .sum::<f64>()
        })
        .collect()
}
pub fn close(got: &[f32], want: &[f64], k: usize) -> Result<(), String> {
    if got.len() != want.len() {
        return Err("GOAL_NOT_MET: output length differs".into());
    }
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        // Bounded fixtures have magnitudes <= 1 and k <= 1003; allow reduction regrouping.
        let tol = 2e-6 * k.max(1) as f64 + 2e-5 * b.abs();
        if !a.is_finite() || !b.is_finite() || (a as f64 - b).abs() > tol {
            return Err(format!(
                "GOAL_NOT_MET: output {i}: got {a}, oracle {b}, tolerance {tol}"
            ));
        }
    }
    Ok(())
}
pub type Kernel = fn(&[f32], &[f32], &[f32], &mut [f32], Shape) -> Result<(), String>;
pub fn verify(kernel: Kernel) -> Result<(), String> {
    for s in [
        Shape { m: 2, k: 3, n: 2 },
        Shape { m: 3, k: 5, n: 7 },
        Shape { m: 1, k: 17, n: 3 },
        Shape { m: 7, k: 33, n: 1 },
        Shape { m: 5, k: 65, n: 9 },
    ] {
        let (x, w, bias) = (
            values(s.m * s.k, 11),
            values(s.n * s.k, 47),
            values(s.n, 19),
        );
        let mut y = vec![99.; s.m * s.n];
        for _ in 0..2 {
            kernel(&x, &w, &bias, &mut y, s)?;
            close(&y, &oracle(&x, &w, &bias, s), s.k)?;
        }
    }
    Ok(())
}
#[derive(Debug)]
pub struct Timing {
    pub samples: Vec<Duration>,
    pub warmups: usize,
}
impl Timing {
    pub fn report(&mut self) -> Result<(), String> {
        if self.samples.is_empty() {
            return Err("no measured samples".into());
        }
        self.samples.sort_unstable();
        println!(
            "warmups={} samples={} median={:?} range={:?}..{:?}",
            self.warmups,
            self.samples.len(),
            self.samples[self.samples.len() / 2],
            self.samples[0],
            self.samples[self.samples.len() - 1]
        );
        Ok(())
    }
}
pub fn benchmark(name: &str, kernel: Kernel) -> Result<(), String> {
    benchmark_with(name, kernel, crate::solutions::ch25::measure)
}
pub fn benchmark_with(
    name: &str,
    kernel: Kernel,
    measure: crate::ch25::Measure,
) -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err("benchmark requires cargo run --release".into());
    }
    let s = Shape {
        m: 37,
        k: 65,
        n: 31,
    };
    let (x, w, bias) = (
        values(s.m * s.k, 11),
        values(s.n * s.k, 47),
        values(s.n, 19),
    );
    let mut y = vec![0.; s.m * s.n];
    kernel(&x, &w, &bias, &mut y, s)?;
    close(&y, &oracle(&x, &w, &bias, s), s.k)?;
    println!("{name}: X=[{},{}] W=[{},{}] f32 arch={} OS={} release; record rustc -Vv, compiler flags and CPU separately",s.m,s.k,s.n,s.k,std::env::consts::ARCH,std::env::consts::OS);
    let mut run = || {
        kernel(
            black_box(&x),
            black_box(&w),
            black_box(&bias),
            black_box(&mut y),
            s,
        )
    };
    let mut timing = measure(&mut run, 3, 11)?;
    timing.report()?;
    let seconds = timing.samples[timing.samples.len() / 2].as_secs_f64();
    println!("work={} FLOPs throughput={:.3} GFLOP/s checksum={:.6}; call includes validation/reset and any thread setup; output allocation/I/O excluded",2*s.m*s.k*s.n,2.*s.m as f64*s.k as f64*s.n as f64/seconds/1e9,black_box(&y).iter().map(|v|*v as f64).sum::<f64>());
    // Separately measure allocation + validation + kernel, without printing inside either interval.
    let mut e2e = measure(
        &mut || {
            let mut output = vec![0.; s.m * s.n];
            kernel(&x, &w, &bias, &mut output, s)?;
            black_box(output);
            Ok(())
        },
        3,
        11,
    )?;
    print!("end-to-end output allocation + call + drop (inputs resident): ");
    e2e.report()?;
    let bytes = 4 * (x.len() + w.len() + bias.len() + y.len());
    println!(
        "cold-array estimate={bytes} bytes intensity={:.4} FLOP/byte; not measured DRAM traffic",
        2. * s.m as f64 * s.k as f64 * s.n as f64 / bytes as f64
    );
    Ok(())
}
pub fn demo(kernel: Kernel) -> Result<(), String> {
    let mut y = [0.; 4];
    kernel(
        &[1., 2., 3., 4., 5., 6.],
        &[1., 3., 5., 2., 4., 6.],
        &[0., 0.],
        &mut y,
        Shape { m: 2, k: 3, n: 2 },
    )?;
    println!("X=[[1,2,3],[4,5,6]], W=[[1,3,5],[2,4,6]], bias=[0,0]; Y={y:?}");
    close(&y, &[22., 28., 49., 64.], 3)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_and_boundaries() -> Result<(), String> {
        demo(dense)?;
        verify(dense)?;
        let s = Shape { m: 1, k: 1, n: 1 };
        assert!(dense(&[f32::NAN], &[1.], &[0.], &mut [0.], s).is_err());
        assert!(dense(&[f32::MAX], &[2.], &[0.], &mut [0.], s).is_err());
        for bad in [
            Shape { m: 0, ..s },
            Shape {
                m: usize::MAX,
                k: 2,
                ..s
            },
            Shape { m: 2, ..s },
        ] {
            assert!(dense(&[1.], &[1.], &[0.], &mut [0.], bad).is_err());
        }
        Ok(())
    }
}
