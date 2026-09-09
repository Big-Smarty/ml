//! Validate once at each public boundary, detect all CPU features, and keep every load within a complete vector.
use crate::{
    ch28,
    common::{self, Shape},
};
pub fn lane_sums(a: &[f32], b: &[f32], width: usize) -> Result<(Vec<f32>, f32), String> {
    ch28::validate(a, b)?;
    if ![4, 8, 16].contains(&width) {
        return Err("lane width must be 4, 8 or 16".into());
    }
    let prefix = a.len() / width * width;
    let mut lanes = vec![0.; width];
    for (left, right) in a[..prefix]
        .chunks_exact(width)
        .zip(b[..prefix].chunks_exact(width))
    {
        for ((sum, x), y) in lanes.iter_mut().zip(left).zip(right) {
            *sum += x * y;
        }
    }
    let tail = ch28::scalar(&a[prefix..], &b[prefix..]);
    common::finite(&lanes)?;
    common::finite(&[tail])?;
    Ok((lanes, tail))
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_kernel(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while a.len() - i >= 8 {
        // SAFETY: caller checked both CPU features and equal lengths; at least 8 elements remain in each slice. loadu requires no alignment promise.
        let (left, right) = unsafe {
            (
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
            )
        };
        acc = _mm256_fmadd_ps(left, right, acc);
        i += 8;
    }
    let mut lanes = [0.; 8];
    // SAFETY: local array has exactly eight writable f32 values; AVX2/FMA is established by caller.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    lanes.iter().sum::<f32>() + ch28::scalar(&a[i..], &b[i..])
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_kernel(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while a.len() - i >= 16 {
        // SAFETY: caller checked AVX-512F and equal lengths; complete 16-value prefix bounds both unaligned loads.
        let (left, right) = unsafe {
            (
                _mm512_loadu_ps(a.as_ptr().add(i)),
                _mm512_loadu_ps(b.as_ptr().add(i)),
            )
        };
        acc = _mm512_add_ps(acc, _mm512_mul_ps(left, right));
        i += 16;
    }
    let mut lanes = [0.; 16];
    // SAFETY: array has 16 writable values and caller established AVX-512F.
    unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), acc) };
    lanes.iter().sum::<f32>() + ch28::scalar(&a[i..], &b[i..])
}
pub fn dot(a: &[f32], b: &[f32]) -> Result<(&'static str, f32), String> {
    ch28::validate(a, b)?;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: validation proved equal finite slices; both required features are detected before the private call.
        let value = unsafe { avx2_kernel(a, b) };
        common::finite(&[value])?;
        return Ok(("avx2+fma", value));
    }
    let value = ch28::scalar(a, b);
    common::finite(&[value])?;
    Ok(("scalar", value))
}
pub fn avx512(a: &[f32], b: &[f32]) -> Result<(&'static str, f32), String> {
    ch28::validate(a, b)?;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: equal lengths are validated and the specific required feature is detected.
        let value = unsafe { avx512_kernel(a, b) };
        common::finite(&[value])?;
        return Ok(("avx512f", value));
    }
    Err(
        "unsupported: AVX-512F is unavailable on this CPU; default dispatch remains portable"
            .into(),
    )
}
/// This independent output update is an auto-vectorization candidate, not an assembly guarantee.
#[inline(never)]
pub fn scale_add(x: &[f32], y: &mut [f32], scale: f32) -> Result<(), String> {
    ch28::validate(x, y)?;
    if !scale.is_finite() {
        return Err("scale must be finite".into());
    }
    for (out, &input) in y.iter_mut().zip(x) {
        *out += scale * input;
    }
    common::finite(y)
}
pub fn dense(x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32], s: Shape) -> Result<(), String> {
    ch28::dense_with(dot, x, w, bias, y, s)
}
pub fn run(args: &[String]) -> Result<(), String> {
    common::demo(dense)?;
    println!("dot backend={}", dot(&[1., 2., 3.], &[4., -1., 0.5])?.0);
    let mut y = [2.; 19];
    scale_add(&[1.; 19], &mut y, 0.5)?;
    println!(
        "auto-vectorization candidate checksum={} (inspect assembly)",
        y.iter().sum::<f32>()
    );
    if args.iter().any(|a| a == "--avx512") {
        println!("{:?}", avx512(&[1.; 19], &[0.5; 19])?);
    }
    if args.iter().any(|a| a == "--bench") {
        common::benchmark(
            "SIMD dense threads=1; includes repeated feature/finite checks",
            dense,
        )?;
    }
    Ok(())
}
pub fn check() -> Result<(), String> {
    ch28::verify(dot, lane_sums, avx512)
}
