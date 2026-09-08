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
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    batch: usize,
    input: usize,
    output: usize,
    y: &mut [f32],
) -> Result<(), String> {
    if batch == 0
        || input == 0
        || output == 0
        || x.len() != batch.checked_mul(input).ok_or("shape overflow")?
        || w.len() != output.checked_mul(input).ok_or("shape overflow")?
        || bias.len() != output
        || y.len() != batch.checked_mul(output).ok_or("shape overflow")?
    {
        return Err("dense shapes must be x=[batch,input], w=[output,input], bias=[output], y=[batch,output]".into());
    }
    for b in 0..batch {
        for o in 0..output {
            let mut sum = bias[o];
            for i in 0..input {
                sum += x[b * input + i] * w[o * input + i];
            }
            y[b * output + o] = sum;
        }
    }
    Ok(())
}

fn dense_f64_reference(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    batch: usize,
    input: usize,
    output: usize,
) -> Vec<f64> {
    let mut y = vec![0.0; batch * output];
    for b in 0..batch {
        for o in 0..output {
            let mut sum = bias[o] as f64;
            for i in 0..input {
                sum += x[b * input + i] as f64 * w[o * input + i] as f64;
            }
            y[b * output + o] = sum;
        }
    }
    y
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn benchmark(batch: usize, input: usize, output: usize, repetitions: usize) -> Result<(), String> {
    let x = values(batch * input, 17);
    let w = values(output * input, 43);
    let bias = values(output, 71);
    let mut y = vec![0.0; batch * output];
    dense(&x, &w, &bias, batch, input, output, &mut y)?;
    let reference = dense_f64_reference(&x, &w, &bias, batch, input, output);
    if y.iter().zip(&reference).any(|(&a, &b)| {
        !a.is_finite() || !b.is_finite() || (a as f64 - b).abs() > 1e-5 + 2e-6 * b.abs()
    }) {
        return Err("f32 kernel disagrees with f64 reference".into());
    }
    for _ in 0..3 {
        dense(
            black_box(&x),
            black_box(&w),
            black_box(&bias),
            batch,
            input,
            output,
            black_box(&mut y),
        )?;
    }
    let mut samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let start = Instant::now();
        dense(
            black_box(&x),
            black_box(&w),
            black_box(&bias),
            batch,
            input,
            output,
            black_box(&mut y),
        )?;
        samples.push(start.elapsed());
    }
    let min = *samples.iter().min().unwrap();
    let max = *samples.iter().max().unwrap();
    let med = median(&mut samples);
    let operations = 2.0 * batch as f64 * input as f64 * output as f64;
    let gflops = operations / med.as_secs_f64() / 1e9;
    let compulsory_bytes = 4.0 * (x.len() + w.len() + bias.len() + y.len()) as f64;
    let intensity = operations / compulsory_bytes;
    let checksum: f64 = black_box(&y).iter().map(|&v| v as f64).sum();
    println!(
        "architecture={} OS={} threads=1",
        env::consts::ARCH,
        env::consts::OS
    );
    println!("dense [{batch},{input}] x [{output},{input}]^T; f32; release mode recommended");
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
        let mut y = [0.0];
        dense(
            &[1.0, 2.0, 3.0],
            &[0.5, -1.0, 0.25],
            &[0.1],
            1,
            3,
            1,
            &mut y,
        )
        .unwrap();
        assert!((y[0] + 0.65).abs() < 1e-6);
    }

    #[test]
    fn f32_kernel_agrees_with_f64_reference() {
        let (batch, input, output) = (3, 5, 4);
        let x = values(batch * input, 17);
        let w = values(output * input, 43);
        let bias = values(output, 71);
        let mut got = vec![0.0; batch * output];
        dense(&x, &w, &bias, batch, input, output, &mut got).unwrap();
        let want = dense_f64_reference(&x, &w, &bias, batch, input, output);
        for (a, b) in got.iter().zip(want) {
            assert!((*a as f64 - b).abs() <= 1e-5 + 2e-6 * b.abs());
        }
    }
    #[test]
    fn bad_shapes_are_rejected() {
        assert!(dense(&[1.0], &[1.0], &[0.0], 2, 1, 1, &mut [0.0]).is_err());
    }
}
