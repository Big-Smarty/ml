//! Working dense baseline. Replace the complete traversal with clipped row/inner/column tiles.
use crate::common::{self, Shape};
/// B is W-transpose, contiguous [k,n]. Returns the number of tiles actually visited.
pub fn blocked(
    x: &[f32],
    b: &[f32],
    bias: &[f32],
    y: &mut [f32],
    s: Shape,
    block: usize,
) -> Result<usize, String> {
    common::validate(x, b, bias, y, s)?;
    if block == 0 {
        return Err("block must be positive".into());
    }
    for row in 0..s.m {
        for col in 0..s.n {
            let mut sum = bias[col];
            for inner in 0..s.k {
                sum += x[row * s.k + inner] * b[inner * s.n + col];
            }
            y[row * s.n + col] = sum;
        }
    }
    common::finite(y)?;
    Ok(1)
}
pub fn dense(x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32], s: Shape) -> Result<(), String> {
    common::validate(x, w, bias, y, s)?;
    let b = transpose(w, s);
    blocked(x, &b, bias, y, s, 4)?;
    Ok(())
}
pub fn transpose(w: &[f32], s: Shape) -> Vec<f32> {
    (0..s.k * s.n)
        .map(|i| w[(i % s.n) * s.k + i / s.n])
        .collect()
}
pub type Blocked = fn(&[f32], &[f32], &[f32], &mut [f32], Shape, usize) -> Result<usize, String>;
pub fn verify(kernel: Blocked) -> Result<(), String> {
    for s in [
        Shape { m: 3, k: 5, n: 7 },
        Shape { m: 1, k: 17, n: 3 },
        Shape { m: 5, k: 3, n: 1 },
    ] {
        let (x, w, bias) = (
            common::values(s.m * s.k, 3),
            common::values(s.n * s.k, 5),
            common::values(s.n, 7),
        );
        let b = transpose(&w, s);
        let mut y = vec![99.; s.m * s.n];
        for block in [1, 4, 16, usize::MAX] {
            for _ in 0..2 {
                let visited = kernel(&x, &b, &bias, &mut y, s, block)?;
                common::close(&y, &common::oracle(&x, &w, &bias, s), s.k)?;
                let expected = s.m.div_ceil(block) * s.k.div_ceil(block) * s.n.div_ceil(block);
                if visited != expected {
                    return Err(format!("GOAL_REVIEW_REQUIRED: shape {s:?} block {block}: visited {visited} tiles, expected {expected}; implement the full blocked traversal and count visits"));
                }
            }
        }
    }
    if kernel(
        &[1.],
        &[1.],
        &[0.],
        &mut [0.],
        Shape { m: 1, k: 1, n: 1 },
        0,
    )
    .is_ok()
    {
        return Err("GOAL_NOT_MET: zero block accepted".into());
    }
    Ok(())
}
pub fn experiment(kernel: Blocked) -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err("benchmark requires --release".into());
    }
    let s = Shape {
        m: 37,
        k: 65,
        n: 31,
    };
    let (x, w, bias) = (
        common::values(s.m * s.k, 11),
        common::values(s.n * s.k, 47),
        common::values(s.n, 19),
    );
    let b = transpose(&w, s);
    let mut y = vec![0.; s.m * s.n];
    for block in [4, 16, 32] {
        kernel(&x, &b, &bias, &mut y, s, block)?;
        common::close(&y, &common::oracle(&x, &w, &bias, s), s.k)?;
        let mut timing = crate::solutions::ch25::measure(
            &mut || {
                std::hint::black_box(kernel(
                    std::hint::black_box(&x),
                    std::hint::black_box(&b),
                    &bias,
                    std::hint::black_box(&mut y),
                    s,
                    block,
                )?);
                Ok(())
            },
            3,
            11,
        )?;
        print!("GEMM+bias X=[{},{}] B=[{},{}] f32 threads=1 block={block}; arch={} OS={} release; transpose/allocation excluded; validation/reset included: ",s.m,s.k,s.k,s.n,std::env::consts::ARCH,std::env::consts::OS);
        timing.report()?;
        println!(
            "GFLOP/s={:.3} checksum={:.6}",
            2. * s.m as f64 * s.k as f64 * s.n as f64
                / timing.samples[timing.samples.len() / 2].as_secs_f64()
                / 1e9,
            std::hint::black_box(&y)
                .iter()
                .map(|v| *v as f64)
                .sum::<f64>()
        );
    }
    Ok(())
}
pub fn run(args: &[String]) -> Result<(), String> {
    common::demo(dense)?;
    if args.iter().any(|a| a == "--bench") {
        experiment(blocked)?;
    }
    Ok(())
}
pub fn check() -> Result<(), String> {
    verify(blocked)?;
    Err("GOAL_REVIEW_REQUIRED: numerical and tile-count checks pass; inspect three clipped tile loops, contiguous inner column traversal, and exactly one bias initialization before claiming locality work complete".into())
}
#[cfg(test)]
mod tests {
    #[test]
    fn baseline_and_solution() -> Result<(), String> {
        super::run(&[])?;
        crate::common::verify(super::dense)?;
        crate::solutions::ch26::check()
    }
}
