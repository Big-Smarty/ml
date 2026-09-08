//! Full and tiled exact causal grouped-query attention on the CPU.
use std::{env, hint::black_box, time::Instant};

#[derive(Clone, Copy)]
struct Shape {
    tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

fn validate(q: &[f32], k: &[f32], v: &[f32], s: Shape) -> Result<(), &'static str> {
    if s.tokens == 0
        || s.query_heads == 0
        || s.kv_heads == 0
        || s.head_dim == 0
        || !s.query_heads.is_multiple_of(s.kv_heads)
    {
        return Err("dimensions must be positive and query_heads divisible by kv_heads");
    }
    let q_len = s
        .tokens
        .checked_mul(s.query_heads)
        .and_then(|x| x.checked_mul(s.head_dim))
        .ok_or("attention shape overflow")?;
    let kv_len = s
        .tokens
        .checked_mul(s.kv_heads)
        .and_then(|x| x.checked_mul(s.head_dim))
        .ok_or("attention shape overflow")?;
    if q.len() != q_len || k.len() != kv_len || v.len() != k.len() {
        return Err("Q, K, or V shape mismatch");
    }
    if q.iter().chain(k).chain(v).any(|x| !x.is_finite()) {
        return Err("attention inputs must be finite");
    }
    Ok(())
}

fn offset(token: usize, head: usize, lane: usize, heads: usize, dim: usize) -> usize {
    (token * heads + head) * dim + lane
}
fn score(q: &[f32], k: &[f32], qi: usize, kj: usize, qh: usize, kh: usize, s: Shape) -> f32 {
    let dot: f32 = (0..s.head_dim)
        .map(|d| {
            q[offset(qi, qh, d, s.query_heads, s.head_dim)]
                * k[offset(kj, kh, d, s.kv_heads, s.head_dim)]
        })
        .sum();
    dot / (s.head_dim as f32).sqrt()
}

fn full_attention(q: &[f32], k: &[f32], v: &[f32], s: Shape) -> Result<Vec<f32>, &'static str> {
    validate(q, k, v, s)?;
    let mut out = vec![0.0; q.len()];
    for qi in 0..s.tokens {
        for qh in 0..s.query_heads {
            let kh = qh / (s.query_heads / s.kv_heads);
            let scores: Vec<f32> = (0..=qi).map(|kj| score(q, k, qi, kj, qh, kh, s)).collect();
            if scores.iter().any(|x| !x.is_finite()) {
                return Err("attention score overflowed");
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = scores.iter().map(|x| (*x - max).exp()).sum();
            for (kj, x) in scores.into_iter().enumerate() {
                let p = (x - max).exp() / denom;
                for d in 0..s.head_dim {
                    out[offset(qi, qh, d, s.query_heads, s.head_dim)] +=
                        p * v[offset(kj, kh, d, s.kv_heads, s.head_dim)];
                }
            }
        }
    }
    if out.iter().any(|x| !x.is_finite()) {
        return Err("attention output is nonfinite");
    }
    Ok(out)
}

fn tiled_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    s: Shape,
    tile: usize,
) -> Result<Vec<f32>, &'static str> {
    validate(q, k, v, s)?;
    if tile == 0 {
        return Err("tile size must be positive");
    }
    let mut out = vec![0.0; q.len()];
    for qi in 0..s.tokens {
        for qh in 0..s.query_heads {
            let kh = qh / (s.query_heads / s.kv_heads);
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0;
            let mut numerator = vec![0.0; s.head_dim];
            for start in (0..=qi).step_by(tile) {
                let end = start.saturating_add(tile).min(qi + 1);
                let tile_scores: Vec<f32> = (start..end)
                    .map(|kj| score(q, k, qi, kj, qh, kh, s))
                    .collect();
                if tile_scores.iter().any(|x| !x.is_finite()) {
                    return Err("attention score overflowed");
                }
                let tile_max = tile_scores
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let next_m = m.max(tile_max);
                let old_scale = (m - next_m).exp();
                numerator.iter_mut().for_each(|x| *x *= old_scale);
                l *= old_scale;
                for (local, x) in tile_scores.into_iter().enumerate() {
                    let weight = (x - next_m).exp();
                    let kj = start + local;
                    l += weight;
                    for d in 0..s.head_dim {
                        numerator[d] += weight * v[offset(kj, kh, d, s.kv_heads, s.head_dim)];
                    }
                }
                m = next_m;
            }
            for d in 0..s.head_dim {
                out[offset(qi, qh, d, s.query_heads, s.head_dim)] = numerator[d] / l;
            }
        }
    }
    if out.iter().any(|x| !x.is_finite()) {
        return Err("attention output is nonfinite");
    }
    Ok(out)
}

fn fixture(s: Shape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let make = |n: usize, mul: usize| {
        (0..n)
            .map(|i| (((i * mul + 7) % 29) as f32 - 14.0) / 11.0)
            .collect()
    };
    (
        make(s.tokens * s.query_heads * s.head_dim, 5),
        make(s.tokens * s.kv_heads * s.head_dim, 9),
        make(s.tokens * s.kv_heads * s.head_dim, 13),
    )
}
fn max_error(a: &[f32], b: &[f32]) -> Result<f32, &'static str> {
    if a.is_empty() || a.len() != b.len() || a.iter().chain(b).any(|x| !x.is_finite()) {
        return Err("comparison needs equal finite slices");
    }
    Ok(a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max))
}

fn bench() -> Result<(), &'static str> {
    let s = Shape {
        tokens: 128,
        query_heads: 8,
        kv_heads: 2,
        head_dim: 16,
    };
    let (q, k, v) = fixture(s);
    for _ in 0..5 {
        black_box(full_attention(&q, &k, &v, s)?);
        black_box(tiled_attention(&q, &k, &v, s, 16)?);
    }
    let mut full_samples = Vec::new();
    let mut tiled_samples = Vec::new();
    let mut full = Vec::new();
    let mut tiled = Vec::new();
    for _ in 0..7 {
        let a = Instant::now();
        full = black_box(full_attention(&q, &k, &v, s)?);
        full_samples.push(a.elapsed());
        let a = Instant::now();
        tiled = black_box(tiled_attention(&q, &k, &v, s, 16)?);
        tiled_samples.push(a.elapsed());
    }
    full_samples.sort();
    tiled_samples.sort();
    let full_t = full_samples[3];
    let tiled_t = tiled_samples[3];
    println!("architecture={} OS={} threads=1 f32 tokens={} query_heads={} kv_heads={} head_dim={} tile=16 warmups=5 samples=7",
        env::consts::ARCH, env::consts::OS, s.tokens, s.query_heads, s.kv_heads, s.head_dim);
    println!("full median={full_t:?} range={:?}..{:?}; tiled median={tiled_t:?} range={:?}..{:?}; max_abs_error={:.8}",
        full_samples[0], full_samples[6], tiled_samples[0], tiled_samples[6], max_error(&full, &tiled)?);
    println!("End-to-end function latency includes validation, output allocation and temporary allocation; this is not an allocation-free kernel benchmark.");
    println!("Local scalar CPU measurements; tiling here proves the recurrence, not a speedup.");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.as_slice() {
        [] => (),
        [flag] if flag == "--bench" => return bench().map_err(Into::into),
        _ => return Err("usage: ch42 [--bench]".into()),
    }
    let s = Shape {
        tokens: 5,
        query_heads: 4,
        kv_heads: 2,
        head_dim: 3,
    };
    let (q, k, v) = fixture(s);
    let full = full_attention(&q, &k, &v, s)?;
    let tiled = tiled_attention(&q, &k, &v, s, 2)?;
    println!("GQA mapping: query heads 0,1 -> KV 0; heads 2,3 -> KV 1");
    println!("max |full - tiled| = {:.8}", max_error(&full, &tiled)?);
    let score_bytes = s.tokens * s.tokens * s.query_heads * 4;
    println!("materialized score estimate: {score_bytes} bytes; tiled implementation materializes at most {} scores per row",2);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiled_matches_oracle_across_tiles() {
        let s = Shape {
            tokens: 7,
            query_heads: 4,
            kv_heads: 2,
            head_dim: 3,
        };
        let (q, k, v) = fixture(s);
        let oracle = full_attention(&q, &k, &v, s).unwrap();
        for tile in [1, 2, 4, 9] {
            let tiled = tiled_attention(&q, &k, &v, s, tile).unwrap();
            for (&expected, &actual) in oracle.iter().zip(&tiled) {
                assert!((expected - actual).abs() <= 2e-6 + 2e-6 * expected.abs());
            }
        }
        assert!(tiled_attention(&q, &k, &v, s, usize::MAX).is_ok());
    }
    #[test]
    fn causal_first_token_equals_its_value() {
        let s = Shape {
            tokens: 2,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 2,
        };
        let q = vec![0.0; 8];
        let k = vec![0.0; 4];
        let v = vec![3.0, 4.0, 9.0, 10.0];
        let o = tiled_attention(&q, &k, &v, s, 1).unwrap();
        assert_eq!(&o[..4], &[3.0, 4.0, 3.0, 4.0]);
        let grouped = Shape {
            tokens: 1,
            query_heads: 4,
            kv_heads: 2,
            head_dim: 2,
        };
        let result = tiled_attention(&[0.0; 8], &[0.0; 4], &v, grouped, 1).unwrap();
        assert_eq!(result, [3.0, 4.0, 3.0, 4.0, 9.0, 10.0, 9.0, 10.0]);
    }
    #[test]
    fn increasing_maxima_and_invalid_numbers() {
        let s = Shape {
            tokens: 3,
            query_heads: 1,
            kv_heads: 1,
            head_dim: 1,
        };
        let q = [1.0; 3];
        let k = [1.0, 3.0, 2.0];
        let v = [2.0, -1.0, 4.0];
        let oracle = full_attention(&q, &k, &v, s).unwrap();
        let expected = ((-2.0_f32).exp() * 2.0 - 1.0 + (-1.0_f32).exp() * 4.0)
            / ((-2.0_f32).exp() + 1.0 + (-1.0_f32).exp());
        assert!((oracle[2] - expected).abs() < 2e-6);
        assert!(max_error(&oracle, &tiled_attention(&q, &k, &v, s, 1).unwrap()).unwrap() < 2e-6);
        for tiled in [false, true] {
            let run = |q: &[f32], k: &[f32]| {
                if tiled {
                    tiled_attention(q, k, &v, s, 1)
                } else {
                    full_attention(q, k, &v, s)
                }
            };
            assert!(run(&[f32::NAN; 3], &k).is_err());
            assert!(run(&[f32::MAX; 3], &[2.0; 3]).is_err());
        }
        assert!(tiled_attention(&q, &k, &v, s, 0).is_err());
    }
    #[test]
    fn rejects_invalid_gqa() {
        let s = Shape {
            tokens: 1,
            query_heads: 3,
            kv_heads: 2,
            head_dim: 1,
        };
        assert!(full_attention(&[0.0; 3], &[0.0; 2], &[0.0; 2], s).is_err());
        let overflow = Shape {
            tokens: usize::MAX,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 2,
        };
        assert!(full_attention(&[], &[], &[], overflow).is_err());
        assert!(max_error(&[1.0], &[]).is_err());
    }
}
