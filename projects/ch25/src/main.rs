//! An allocation-free dense-inference kernel with an honest microbenchmark.
use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

fn values(len: usize, salt: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            (((i as u32).wrapping_mul(1664525).wrapping_add(salt) % 2001) as f32 - 1000.0) / 1000.0
        })
        .collect()
}

fn dense(
    inputs: &[f32],
    weights: &[f32],
    bias: &[f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
    outputs: &mut [f32],
) -> Result<(), String> {
    if batch == 0
        || in_features == 0
        || out_features == 0
        || inputs.len() != batch.checked_mul(in_features).ok_or("shape overflow")?
        || weights.len()
            != out_features
                .checked_mul(in_features)
                .ok_or("shape overflow")?
        || bias.len() != out_features
        || outputs.len() != batch.checked_mul(out_features).ok_or("shape overflow")?
    {
        return Err("dense shapes must be inputs=[batch,in_features], weights=[out_features,in_features], bias=[out_features], outputs=[batch,out_features]".into());
    }
    for b in 0..batch {
        for o in 0..out_features {
            let mut sum = bias[o];
            for i in 0..in_features {
                sum += inputs[b * in_features + i] * weights[o * in_features + i];
            }
            outputs[b * out_features + o] = sum;
        }
    }
    Ok(())
}

fn dense_f64_reference(
    inputs: &[f32],
    weights: &[f32],
    bias: &[f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f64> {
    let mut outputs = vec![0.0; batch * out_features];
    for b in 0..batch {
        for o in 0..out_features {
            let mut sum = bias[o] as f64;
            for i in 0..in_features {
                sum += inputs[b * in_features + i] as f64 * weights[o * in_features + i] as f64;
            }
            outputs[b * out_features + o] = sum;
        }
    }
    outputs
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn benchmark(
    batch: usize,
    in_features: usize,
    out_features: usize,
    repetitions: usize,
) -> Result<(), String> {
    let inputs = values(batch * in_features, 17);
    let weights = values(out_features * in_features, 43);
    let bias = values(out_features, 71);
    let mut outputs = vec![0.0; batch * out_features];
    dense(
        &inputs,
        &weights,
        &bias,
        batch,
        in_features,
        out_features,
        &mut outputs,
    )?;
    let reference = dense_f64_reference(&inputs, &weights, &bias, batch, in_features, out_features);
    if outputs.iter().zip(&reference).any(|(&a, &b)| {
        !a.is_finite() || !b.is_finite() || (a as f64 - b).abs() > 1e-5 + 2e-6 * b.abs()
    }) {
        return Err("f32 kernel disagrees with f64 reference".into());
    }
    for _ in 0..3 {
        dense(
            black_box(&inputs),
            black_box(&weights),
            black_box(&bias),
            batch,
            in_features,
            out_features,
            black_box(&mut outputs),
        )?;
    }
    let mut samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let start = Instant::now();
        dense(
            black_box(&inputs),
            black_box(&weights),
            black_box(&bias),
            batch,
            in_features,
            out_features,
            black_box(&mut outputs),
        )?;
        samples.push(start.elapsed());
    }
    let min = *samples.iter().min().unwrap();
    let max = *samples.iter().max().unwrap();
    let med = median(&mut samples);
    let operations = 2.0 * batch as f64 * in_features as f64 * out_features as f64;
    let gflops = operations / med.as_secs_f64() / 1e9;
    let compulsory_bytes = 4.0 * (inputs.len() + weights.len() + bias.len() + outputs.len()) as f64;
    let intensity = operations / compulsory_bytes;
    let checksum: f64 = black_box(&outputs).iter().map(|&v| v as f64).sum();
    println!(
        "architecture={} OS={} threads=1",
        env::consts::ARCH,
        env::consts::OS
    );
    println!("dense inputs=[{batch},{in_features}] weights=[{out_features},{in_features}]; outputs=[{batch},{out_features}]; f32; release mode recommended");
    println!("warmups=3 samples={repetitions} median={med:?} range={min:?}..{max:?}");
    println!("measured throughput={gflops:.3} GFLOP/s checksum={checksum:.6}");
    println!("correctness checked against an f64 accumulator before timing");
    println!("algorithmic work={operations:.0} FLOPs; cold-array traffic estimate={compulsory_bytes:.0} bytes; intensity under that assumption={intensity:.2} FLOP/byte (not measured warm-cache DRAM traffic)");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    match args.as_slice() {
        [_] => benchmark(32, 64, 32, 11)?,
        [_, flag] if flag == "--large" => benchmark(256, 512, 256, 21)?,
        _ => return Err("usage: ch25 [--large]".into()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_matches_hand_calculation() {
        let mut outputs = [0.0];
        dense(
            &[1.0, 2.0, 3.0],
            &[0.5, -1.0, 0.25],
            &[0.1],
            1,
            3,
            1,
            &mut outputs,
        )
        .unwrap();
        assert!((outputs[0] + 0.65).abs() < 1e-6);
    }

    #[test]
    fn f32_kernel_agrees_with_f64_reference() {
        let (batch, in_features, out_features) = (3, 5, 4);
        let inputs = values(batch * in_features, 17);
        let weights = values(out_features * in_features, 43);
        let bias = values(out_features, 71);
        let mut got = vec![0.0; batch * out_features];
        dense(
            &inputs,
            &weights,
            &bias,
            batch,
            in_features,
            out_features,
            &mut got,
        )
        .unwrap();
        let want = dense_f64_reference(&inputs, &weights, &bias, batch, in_features, out_features);
        for (a, b) in got.iter().zip(want) {
            assert!((*a as f64 - b).abs() <= 1e-5 + 2e-6 * b.abs());
        }
    }
    #[test]
    fn bad_shapes_are_rejected() {
        assert!(dense(&[1.0], &[1.0], &[0.0], 2, 1, 1, &mut [0.0]).is_err());
    }
}
