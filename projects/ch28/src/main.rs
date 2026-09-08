//! Portable scalar dot product with guarded AVX-512 and AVX2/FMA implementations.
use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn dot_f64_reference(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

#[inline(never)]
fn scale_add_auto(x: &[f32], y: &mut [f32], scale: f32) {
    for (i, v) in y.iter_mut().enumerate() {
        *v += scale * x[i];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= a.len() {
        // SAFETY: dispatch establishes AVX2+FMA; the loop proves both unaligned 8-lane reads are in bounds.
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(av, bv, acc);
        i += 8;
    }
    let mut lanes = [0.; 8];
    // SAFETY: this local array has eight writable lanes; the caller established AVX2+FMA.
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    lanes.iter().sum::<f32>() + dot_scalar(&a[i..], &b[i..])
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i + 16 <= a.len() {
        // SAFETY: dispatch establishes AVX-512F; the loop proves both unaligned 16-lane reads are in bounds.
        let av = _mm512_loadu_ps(a.as_ptr().add(i));
        let bv = _mm512_loadu_ps(b.as_ptr().add(i));
        acc = _mm512_add_ps(acc, _mm512_mul_ps(av, bv));
        i += 16;
    }
    let mut lanes = [0.; 16];
    // SAFETY: this local array has sixteen writable lanes; the caller established AVX-512F.
    _mm512_storeu_ps(lanes.as_mut_ptr(), acc);
    lanes.iter().sum::<f32>() + dot_scalar(&a[i..], &b[i..])
}

fn dot_dispatch(a: &[f32], b: &[f32]) -> Result<(&'static str, f32), String> {
    if a.len() != b.len() {
        return Err("dot-product vectors must have equal lengths".into());
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: both target features are checked and equal lengths were checked above.
            return Ok(("avx2+fma", unsafe { dot_avx2_fma(a, b) }));
        }
    }
    Ok(("scalar", dot_scalar(a, b)))
}

fn dot_avx512_checked(a: &[f32], b: &[f32]) -> Result<(&'static str, f32), String> {
    if a.len() != b.len() {
        return Err("dot-product vectors must have equal lengths".into());
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        // SAFETY: the runtime check satisfies dot_avx512's only CPU-feature precondition; equal lengths were checked above.
        return Ok(("avx512f", unsafe { dot_avx512(a, b) }));
    }
    Err("AVX-512F is not supported on this CPU; use the portable default".into())
}

fn values(n: usize, salt: u32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i as u32 * 73 + salt) % 257) as f32 - 128.) / 128.)
        .collect()
}
fn median(v: &mut [Duration]) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}
fn experiment(n: usize, repetitions: usize, request_avx512: bool) -> Result<(), String> {
    let a = values(n, 11);
    let b = values(n, 47);
    let mut y = vec![1.; n];
    scale_add_auto(&a, &mut y, 0.25);
    let run = |a: &[f32], b: &[f32]| {
        if request_avx512 {
            dot_avx512_checked(a, b)
        } else {
            dot_dispatch(a, b)
        }
    };
    let reference = dot_f64_reference(&a, &b);
    let (_, checked) = run(&a, &b)?;
    let tolerance = 2e-5 * n as f64 + 2e-5 * reference.abs();
    if !checked.is_finite()
        || !reference.is_finite()
        || (checked as f64 - reference).abs() > tolerance
    {
        return Err("selected SIMD result disagrees with f64 reference".into());
    }
    for _ in 0..3 {
        black_box(run(black_box(&a), black_box(&b))?.1);
    }
    let mut times = Vec::with_capacity(repetitions);
    let mut backend = "";
    let mut answer = 0.;
    for _ in 0..repetitions {
        let t = Instant::now();
        (backend, answer) = run(black_box(&a), black_box(&b))?;
        black_box(answer);
        times.push(t.elapsed());
    }
    let lo = *times.iter().min().unwrap();
    let hi = *times.iter().max().unwrap();
    let med = median(&mut times);
    println!("dot length={n}, backend={backend}, result={answer:.6}; f32; threads=1; architecture={} OS={}", env::consts::ARCH, env::consts::OS);
    println!("timing includes runtime feature dispatch, shape validation, and result handling");
    println!("warmups=3 samples={repetitions} median={med:?} range={lo:?}..{hi:?}");
    println!("auto-vectorization candidate checksum={:.6}; verify generated code before claiming vectorization",black_box(&y).iter().map(|&v|v as f64).sum::<f64>());
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = env::args().collect();
    match a.as_slice() {
        [_] => experiment(1031, 15, false)?,
        [_, f] if f == "--large" => experiment(1_000_003, 21, false)?,
        [_, f] if f == "--avx512" => experiment(1031, 15, true)?,
        [_, f] if f == "--large-avx512" => experiment(1_000_003, 21, true)?,
        _ => return Err("usage: ch28 [--large | --avx512 | --large-avx512]".into()),
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(got: f32, reference: f64, n: usize) -> bool {
        (got as f64 - reference).abs() <= 2e-5 * n as f64 + 2e-5 * reference.abs()
    }
    #[test]
    fn dispatch_handles_empty_short_and_odd_vectors() {
        for n in 0..40 {
            let a = values(n, 11);
            let b = values(n, 47);
            let (_, got) = dot_dispatch(&a, &b).unwrap();
            assert!(close(got, dot_f64_reference(&a, &b), n), "n={n}");
        }
        let n = 1003;
        let a = values(n, 3);
        let b = values(n, 5);
        assert!(close(
            dot_dispatch(&a, &b).unwrap().1,
            dot_f64_reference(&a, &b),
            n
        ));
    }
    #[test]
    fn every_supported_backend_matches_the_oracle() {
        for n in (0..40).chain([1003]) {
            let a = values(n, 3);
            let b = values(n, 5);
            let reference = dot_f64_reference(&a, &b);
            assert!(close(dot_scalar(&a, &b), reference, n));
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // SAFETY: both features are detected and equal-length slices prove all memory preconditions.
                    assert!(close(unsafe { dot_avx2_fma(&a, &b) }, reference, n));
                }
                if is_x86_feature_detected!("avx512f") {
                    // SAFETY: AVX-512F is detected and equal-length slices prove all memory preconditions.
                    assert!(close(unsafe { dot_avx512(&a, &b) }, reference, n));
                }
            }
        }
    }
    #[test]
    fn last_element_is_never_lost_at_a_vector_boundary() {
        for n in [7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let mut a = vec![0.0; n];
            a[n - 1] = 1.0;
            let b = vec![1.0; n];
            assert_eq!(dot_dispatch(&a, &b).unwrap().1, 1.0);
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if is_x86_feature_detected!("avx512f") {
                assert_eq!(dot_avx512_checked(&a, &b).unwrap().1, 1.0);
            }
        }
    }
    #[test]
    fn rejects_length_mismatch() {
        assert!(dot_dispatch(&[1.], &[1., 2.]).is_err());
        assert!(dot_avx512_checked(&[1.], &[1., 2.]).is_err());
    }
    #[test]
    fn auto_kernel_handles_odd_length() {
        let x = values(19, 1);
        let mut y = vec![2.; 19];
        scale_add_auto(&x, &mut y, 0.5);
        for (i, v) in y.iter().enumerate() {
            assert_eq!(*v, 2. + 0.5 * x[i]);
        }
    }
}
