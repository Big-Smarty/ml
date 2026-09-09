//! Full and tiled exact causal grouped-query attention on the CPU.
use std::{env, hint::black_box, time::Instant};

#[derive(Clone, Copy)]
struct Shape {
    positions: usize,
    query_heads: usize,
    kv_heads: usize,
    head_width: usize,
}

fn validate(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: Shape,
) -> Result<(), &'static str> {
    if shape.positions == 0
        || shape.query_heads == 0
        || shape.kv_heads == 0
        || shape.head_width == 0
        || !shape.query_heads.is_multiple_of(shape.kv_heads)
    {
        return Err("dimensions must be positive and query_heads divisible by kv_heads");
    }
    let query_len = shape
        .positions
        .checked_mul(shape.query_heads)
        .and_then(|x| x.checked_mul(shape.head_width))
        .ok_or("attention shape overflow")?;
    let key_value_len = shape
        .positions
        .checked_mul(shape.kv_heads)
        .and_then(|x| x.checked_mul(shape.head_width))
        .ok_or("attention shape overflow")?;
    if queries.len() != query_len || keys.len() != key_value_len || values.len() != keys.len() {
        return Err("Q, K, or V shape mismatch");
    }
    if queries
        .iter()
        .chain(keys)
        .chain(values)
        .any(|x| !x.is_finite())
    {
        return Err("attention inputs must be finite");
    }
    Ok(())
}

fn offset(position: usize, head: usize, lane: usize, heads: usize, head_width: usize) -> usize {
    (position * heads + head) * head_width + lane
}
fn score(
    queries: &[f32],
    keys: &[f32],
    query_position: usize,
    key_position: usize,
    query_head: usize,
    kv_head: usize,
    shape: Shape,
) -> f32 {
    let dot: f32 = (0..shape.head_width)
        .map(|d| {
            queries[offset(
                query_position,
                query_head,
                d,
                shape.query_heads,
                shape.head_width,
            )] * keys[offset(key_position, kv_head, d, shape.kv_heads, shape.head_width)]
        })
        .sum();
    dot / (shape.head_width as f32).sqrt()
}

fn causal_attention_scalar(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: Shape,
) -> Result<Vec<f32>, &'static str> {
    validate(queries, keys, values, shape)?;
    let mut output = vec![0.0; queries.len()];
    for query_position in 0..shape.positions {
        for query_head in 0..shape.query_heads {
            let kv_head = query_head / (shape.query_heads / shape.kv_heads);
            let scores: Vec<f32> = (0..=query_position)
                .map(|key_position| {
                    score(
                        queries,
                        keys,
                        query_position,
                        key_position,
                        query_head,
                        kv_head,
                        shape,
                    )
                })
                .collect();
            if scores.iter().any(|x| !x.is_finite()) {
                return Err("attention score overflowed");
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = scores.iter().map(|score| (*score - maximum).exp()).sum();
            for (key_position, score) in scores.into_iter().enumerate() {
                let probability = (score - maximum).exp() / denominator;
                for lane in 0..shape.head_width {
                    output[offset(
                        query_position,
                        query_head,
                        lane,
                        shape.query_heads,
                        shape.head_width,
                    )] += probability
                        * values[offset(
                            key_position,
                            kv_head,
                            lane,
                            shape.kv_heads,
                            shape.head_width,
                        )];
                }
            }
        }
    }
    if output.iter().any(|x| !x.is_finite()) {
        return Err("attention output is nonfinite");
    }
    Ok(output)
}

fn causal_attention_tiled(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: Shape,
    key_tile_width: usize,
) -> Result<Vec<f32>, &'static str> {
    Ok(tiled_with_peak(queries, keys, values, shape, key_tile_width)?.0)
}
fn tiled_with_peak(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: Shape,
    key_tile_width: usize,
) -> Result<(Vec<f32>, usize), &'static str> {
    // LEARNER: implement bounded key-tile score construction, persistent m/l/o,
    // maximum rescaling and causal GQA output. The scalar oracle stays supplied.
    if key_tile_width == 0 {
        return Err("tile size must be positive");
    }
    Ok((
        causal_attention_scalar(queries, keys, values, shape)?,
        shape.positions,
    ))
}

fn fixture(shape: Shape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let make = |n: usize, mul: usize| {
        (0..n)
            .map(|i| (((i * mul + 7) % 29) as f32 - 14.0) / 11.0)
            .collect()
    };
    (
        make(shape.positions * shape.query_heads * shape.head_width, 5),
        make(shape.positions * shape.kv_heads * shape.head_width, 9),
        make(shape.positions * shape.kv_heads * shape.head_width, 13),
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
    let shape = Shape {
        positions: 128,
        query_heads: 8,
        kv_heads: 2,
        head_width: 16,
    };
    let (queries, keys, values) = fixture(shape);
    for _ in 0..5 {
        black_box(causal_attention_scalar(&queries, &keys, &values, shape)?);
        black_box(causal_attention_tiled(&queries, &keys, &values, shape, 16)?);
    }
    let mut scalar_samples = Vec::new();
    let mut tiled_samples = Vec::new();
    let mut scalar = Vec::new();
    let mut tiled = Vec::new();
    for _ in 0..7 {
        let a = Instant::now();
        scalar = black_box(causal_attention_scalar(&queries, &keys, &values, shape)?);
        scalar_samples.push(a.elapsed());
        let a = Instant::now();
        tiled = black_box(causal_attention_tiled(&queries, &keys, &values, shape, 16)?);
        tiled_samples.push(a.elapsed());
    }
    scalar_samples.sort();
    tiled_samples.sort();
    let scalar_t = scalar_samples[3];
    let tiled_t = tiled_samples[3];
    println!("architecture={} OS={} threads=1 f32 positions={} query_heads={} kv_heads={} head_width={} key_tile_width=16 warmups=5 samples=7",
        env::consts::ARCH, env::consts::OS, shape.positions, shape.query_heads, shape.kv_heads, shape.head_width);
    println!("scalar median={scalar_t:?} range={:?}..{:?}; tiled median={tiled_t:?} range={:?}..{:?}; max_abs_error={:.8}",
        scalar_samples[0], scalar_samples[6], tiled_samples[0], tiled_samples[6], max_error(&scalar, &tiled)?);
    println!("End-to-end function latency includes validation, output allocation and temporary allocation; this is not an allocation-free kernel benchmark.");
    println!("Local scalar CPU measurements; tiling here proves the recurrence, not a speedup.");
    Ok(())
}

fn experiment() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape {
        positions: 5,
        query_heads: 4,
        kv_heads: 2,
        head_width: 3,
    };
    let (queries, keys, values) = fixture(shape);
    let scalar = causal_attention_scalar(&queries, &keys, &values, shape)?;
    let (tiled, peak) = tiled_with_peak(&queries, &keys, &values, shape, 2)?;
    println!("GQA mapping: query heads 0,1 -> KV 0; heads 2,3 -> KV 1");
    println!("max |scalar - tiled| = {:.8}", max_error(&scalar, &tiled)?);
    let score_bytes = shape.positions * shape.positions * shape.query_heads * 4;
    println!("materialized score estimate: {score_bytes} bytes; actual peak score vector per query/head: {} entries",peak);
    Ok(())
}

/// Run the useful baseline; --check separately verifies the learning goal.
pub fn run(args: &[String]) -> Result<(), String> {
    if args == ["--bench"] {
        return bench().map_err(|e| e.to_string());
    }
    if !args.is_empty() {
        return Err("unexpected experiment argument".into());
    }
    experiment().map_err(|e| e.to_string())
}

/// The returned peak counts the scores actually constructed, not a claimed estimate.
fn attention_with_peak(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: Shape,
    tile: usize,
) -> Result<(Vec<f32>, usize), &'static str> {
    tiled_with_peak(q, k, v, shape, tile)
}
pub fn check() -> Result<(), String> {
    let verify = || -> Result<(), &'static str> {
        let shape = Shape {
            positions: 7,
            query_heads: 4,
            kv_heads: 2,
            head_width: 3,
        };
        let (q, k, v) = fixture(shape);
        let oracle = causal_attention_scalar(&q, &k, &v, shape)?;
        for tile in [1, 2, 4, 9] {
            let (out, peak) = attention_with_peak(&q, &k, &v, shape, tile)?;
            if peak > tile
                || out
                    .iter()
                    .zip(&oracle)
                    .any(|(a, b)| (a - b).abs() > 2e-6 + 2e-6 * b.abs())
            {
                return Err("GOAL_NOT_MET: implement online numerator/denominator merge: bounded scores must match every oracle coordinate");
            }
        }
        let s = Shape {
            positions: 3,
            query_heads: 1,
            kv_heads: 1,
            head_width: 1,
        };
        let (out, _) = attention_with_peak(&[1.; 3], &[1001., 1003., 1002.], &[2., -1., 4.], s, 1)?;
        let expected = ((-2.0_f32).exp() * 2. - 1. + (-1.0_f32).exp() * 4.)
            / ((-2.0_f32).exp() + 1. + (-1.0_f32).exp());
        if (out[2] - expected).abs() > 2e-6 {
            return Err("GOAL_NOT_MET: rescale old numerator when the running maximum increases");
        }
        println!("goal: uneven tiles, large scores, causal coverage and GQA oracle agreement pass");
        Ok(())
    };
    verify().map_err(str::to_string)
}

#[cfg(test)]
mod baseline_tests {
    #[test]
    fn supplied_baseline_runs() {
        assert!(super::run(&[]).is_ok());
    }
}
