//! Scalar baseline. Build complete lane accumulation, guarded AVX2/FMA and AVX-512 kernels.
use crate::common::{self, Shape};
pub fn validate(a: &[f32], b: &[f32]) -> Result<(), String> {
    if a.len() != b.len() || a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err("dot inputs must have equal lengths and finite values".into());
    }
    Ok(())
}
pub fn scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
pub fn dot(a: &[f32], b: &[f32]) -> Result<(&'static str, f32), String> {
    validate(a, b)?;
    let value = scalar(a, b);
    common::finite(&[value])?;
    Ok(("scalar", value))
}
/// Portable lane grouping, independent of actual SIMD instruction support.
pub fn lane_sums(a: &[f32], b: &[f32], width: usize) -> Result<(Vec<f32>, f32), String> {
    validate(a, b)?;
    if ![4, 8, 16].contains(&width) {
        return Err("lane width must be 4, 8 or 16".into());
    }
    let mut lanes = vec![0.; width];
    lanes[0] = scalar(a, b);
    Ok((lanes, 0.))
}
pub fn avx512(a: &[f32], b: &[f32]) -> Result<(&'static str, f32), String> {
    validate(a, b)?;
    Err(
        "GOAL_NOT_MET: AVX-512 implementation is the learning extension; baseline remains scalar"
            .into(),
    )
}
pub fn dense(x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32], s: Shape) -> Result<(), String> {
    dense_with(dot, x, w, bias, y, s)
}
pub type Dot = fn(&[f32], &[f32]) -> Result<(&'static str, f32), String>;
pub type Lanes = fn(&[f32], &[f32], usize) -> Result<(Vec<f32>, f32), String>;
pub fn dense_with(
    dot: Dot,
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    y: &mut [f32],
    s: Shape,
) -> Result<(), String> {
    common::validate(x, w, bias, y, s)?;
    // ponytail: checked dispatch repeats per output; move validation outside only if profiling justifies a private prevalidated path.
    for (input, output) in x.chunks_exact(s.k).zip(y.chunks_exact_mut(s.n)) {
        for ((weights, &bias), out) in w.chunks_exact(s.k).zip(bias).zip(output) {
            *out = bias + dot(input, weights)?.1;
        }
    }
    common::finite(y)
}
pub fn verify(dot: Dot, lanes: Lanes, wide: Dot) -> Result<(), String> {
    let a: Vec<_> = (1..=19).map(|v| v as f32).collect();
    let b = vec![0.5; 19];
    let (sums, tail) = lanes(&a, &b, 8)?;
    if sums != [5., 6., 7., 8., 9., 10., 11., 12.] || tail != 27. {
        return Err(format!("GOAL_NOT_MET: expected eight lane sums [5,6,7,8,9,10,11,12] and tail 27; got {sums:?}, tail {tail}; implement complete vectors then tail"));
    }
    for n in (0..40).chain([65, 1003]) {
        let a = common::values(n, 3);
        let b = common::values(n, 5);
        let oracle = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum::<f64>();
        let (backend, value) = dot(&a, &b)?;
        common::close(&[value], &[oracle], n)?;
        common::close(&[scalar(&a, &b)], &[oracle], n)?;
        for width in [4, 8, 16] {
            let (parts, tail) = lanes(&a, &b, width)?;
            common::close(&[parts.iter().sum::<f32>() + tail], &[oracle], n)?;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && backend != "avx2+fma"
            {
                return Err("GOAL_REVIEW_REQUIRED: supported host must exercise guarded AVX2+FMA implementation".into());
            }
            if is_x86_feature_detected!("avx512f") {
                common::close(&[wide(&a, &b)?.1], &[oracle], n)?;
            }
        }
        std::hint::black_box((backend, wide));
    }
    for n in [7, 8, 9, 15, 16, 17, 31, 32, 33] {
        let mut a = vec![0.; n];
        a[n - 1] = 1.;
        let b = vec![1.; n];
        if dot(&a, &b)?.1 != 1. {
            return Err("GOAL_NOT_MET: lost scalar tail impulse".into());
        }
    }
    if dot(&[1.], &[]).is_ok() || dot(&[f32::NAN], &[1.]).is_ok() || dot(&[f32::MAX], &[2.]).is_ok()
    {
        return Err("GOAL_NOT_MET: invalid dot inputs or overflow accepted".into());
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    println!(
        "actual capabilities: AVX2+FMA={}, AVX-512F={}",
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
        is_x86_feature_detected!("avx512f")
    );
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    println!(
        "unsupported on this architecture: x86 SIMD execution; portable lanes/scalar verified"
    );
    println!("SIMD check: portable lanes, forced scalar, dispatch, tails and every hardware-supported wide path verified");
    Ok(())
}
pub fn run(args: &[String]) -> Result<(), String> {
    common::demo(dense)?;
    println!("dot backend={}", dot(&[1., 2., 3.], &[4., -1., 0.5])?.0);
    if args.iter().any(|a| a == "--avx512") {
        println!("{:?}", avx512(&[1.; 19], &[0.5; 19])?);
    }
    if args.iter().any(|a| a == "--bench") {
        common::benchmark("scalar dense threads=1", dense)?;
    }
    Ok(())
}
pub fn check() -> Result<(), String> {
    verify(dot, lane_sums, avx512)?;
    Err("GOAL_REVIEW_REQUIRED: lane and numerical checks pass; review architecture gates, actual intrinsic loop, feature detection, bounds/store SAFETY proofs, and scalar fallback. Backend labels alone do not prove SIMD execution".into())
}
#[cfg(test)]
mod tests {
    #[test]
    fn baselines_and_solution() -> Result<(), String> {
        super::run(&[])?;
        crate::common::verify(super::dense)?;
        crate::common::verify(crate::solutions::ch28::dense)?;
        crate::solutions::ch28::check()
    }
}
