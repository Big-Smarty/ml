//! Symmetric per-output-row int8 and packed-int4 weight quantization.
use ch36::{Config, Decoder};
use std::{hint::black_box, time::Instant};

#[derive(Debug, Clone)]
struct Quantized {
    out_features: usize,
    in_features: usize,
    bits: u8,
    scales: Vec<f32>,
    data: Vec<u8>,
}

fn quantize(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    bits: u8,
) -> Result<Quantized, &'static str> {
    if out_features == 0
        || in_features == 0
        || out_features.checked_mul(in_features) != Some(weights.len())
    {
        return Err("weights must be a nonempty [out_features, in_features] matrix");
    }
    if bits != 8 && bits != 4 {
        return Err("only int8 and int4 are supported");
    }
    if weights.iter().any(|x| !x.is_finite()) {
        return Err("weights must be finite");
    }
    let qmax = if bits == 8 { 127.0 } else { 7.0 };
    let mut scales = Vec::with_capacity(out_features);
    let mut values = Vec::with_capacity(weights.len());
    for row in weights.chunks_exact(in_features) {
        let max_abs = row.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
        let scale = if max_abs == 0.0 {
            1.0
        } else {
            (max_abs / qmax).max(f32::MIN_POSITIVE)
        };
        scales.push(scale);
        values.extend(
            row.iter()
                .map(|x| (x / scale).round().clamp(-qmax, qmax) as i8),
        );
    }
    let data = if bits == 8 {
        values.into_iter().map(|q| q as u8).collect()
    } else {
        values
            .chunks(2)
            .map(|pair| pack_int4_pair(pair[0], pair.get(1).copied().unwrap_or(0)))
            .collect()
    };
    Ok(Quantized {
        out_features,
        in_features,
        bits,
        scales,
        data,
    })
}

fn pack_int4_pair(low: i8, high: i8) -> u8 {
    (low as u8 & 0x0f) | ((high as u8 & 0x0f) << 4)
}

impl Quantized {
    fn value(&self, index: usize) -> i8 {
        if self.bits == 8 {
            self.data[index] as i8
        } else {
            let nibble = if index.is_multiple_of(2) {
                self.data[index / 2] & 0x0f
            } else {
                self.data[index / 2] >> 4
            };
            if nibble & 0x08 != 0 {
                (nibble | 0xf0) as i8
            } else {
                nibble as i8
            }
        }
    }

    fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, &'static str> {
        if input.len() != self.in_features || input.iter().any(|x| !x.is_finite()) {
            return Err("input must be finite and match the input width");
        }
        let output: Vec<f32> = (0..self.out_features)
            .map(|r| {
                (0..self.in_features)
                    .map(|c| {
                        self.value(r * self.in_features + c) as f32 * self.scales[r] * input[c]
                    })
                    .sum()
            })
            .collect();
        if output.iter().any(|x| !x.is_finite()) {
            return Err("matvec output exceeds finite f32 range");
        }
        Ok(output)
    }

    fn storage_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * std::mem::size_of::<f32>()
    }
    fn dequantize(&self) -> Vec<f32> {
        (0..self.out_features * self.in_features)
            .map(|i| self.value(i) as f32 * self.scales[i / self.in_features])
            .collect()
    }
}

fn dense_matvec_reference(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    input: &[f32],
) -> Vec<f32> {
    (0..out_features)
        .map(|r| {
            (0..in_features)
                .map(|c| weights[r * in_features + c] * input[c])
                .sum()
        })
        .collect()
}

fn quantization_error(reference: &[f32], candidate: &[f32]) -> Result<(f32, f32), &'static str> {
    if reference.is_empty()
        || reference.len() != candidate.len()
        || reference.iter().chain(candidate).any(|x| !x.is_finite())
    {
        return Err("quantization error needs equal, nonempty, finite slices");
    }
    let mut sum = 0.0;
    let mut max = 0.0_f32;
    for (a, b) in reference.iter().zip(candidate) {
        let e = (a - b).abs();
        sum += e * e;
        max = max.max(e);
    }
    if !sum.is_finite() || !max.is_finite() {
        return Err("quantization error overflowed f32");
    }
    Ok(((sum / reference.len() as f32).sqrt(), max))
}

fn fixture(out_features: usize, in_features: usize) -> Vec<f32> {
    (0..out_features * in_features)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0)
        .collect()
}

fn bench() -> Result<(), &'static str> {
    println!("CPU architecture={} OS={} f32 activations, 256x256, threads=1, warmups=5, samples=7; allocating end-to-end calls; record CPU model and rustc separately",std::env::consts::ARCH,std::env::consts::OS);
    let (out_features, in_features, repeats) = (256, 256, 200);
    let weights = fixture(out_features, in_features);
    let input: Vec<f32> = (0..in_features).map(|i| (i as f32 * 0.17).sin()).collect();
    let reference = dense_matvec_reference(&weights, out_features, in_features, &input);
    let measure = |f: &mut dyn FnMut() -> Vec<f32>| {
        for _ in 0..5 {
            black_box(f());
        }
        let mut samples = Vec::new();
        let mut checksum = 0.0;
        for _ in 0..7 {
            let start = Instant::now();
            for _ in 0..repeats {
                checksum += black_box(f())[0];
            }
            samples.push(start.elapsed());
        }
        samples.sort();
        (samples[3], samples[0], samples[6], checksum)
    };
    let (dense_t, dense_min, dense_max, dense_sum) = measure(&mut || {
        dense_matvec_reference(
            black_box(&weights),
            out_features,
            in_features,
            black_box(&input),
        )
    });
    println!("fp32 median={dense_t:?} range={dense_min:?}..{dense_max:?} for {repeats} matvecs; checksum={dense_sum:.3}");
    for bits in [8, 4] {
        let q = quantize(&weights, out_features, in_features, bits)?;
        let output = q.matvec(&input)?;
        let (rmse, max_abs) = quantization_error(&reference, &output)?;
        let (time, minimum, maximum, sum) = measure(&mut || q.matvec(black_box(&input)).unwrap());
        println!("int{bits} median={time:?} range={minimum:?}..{maximum:?}; rmse={rmse:.5}, max={max_abs:.5}, checksum={sum:.3}");
    }
    println!(
        "Times are local measurements of this dequantizing scalar kernel, not universal speedups."
    );
    Ok(())
}

fn experiment() -> Result<(), Box<dyn std::error::Error>> {
    let (out_features, in_features) = (3, 6);
    let weights = fixture(out_features, in_features);
    let input = [0.5, -1.0, 0.25, 2.0, -0.5, 1.5];
    let reference = dense_matvec_reference(&weights, out_features, in_features, &input);
    println!(
        "dense output: {reference:?}; fp32 weight bytes={}",
        weights.len() * 4
    );
    for bits in [8, 4] {
        let q = quantize(&weights, out_features, in_features, bits)?;
        let output = q.matvec(&input)?;
        let (rmse, max_abs) = quantization_error(&reference, &output)?;
        println!(
            "int{bits}: output={output:?}, bytes={}, rmse={rmse:.6}, max_abs={max_abs:.6}",
            q.storage_bytes()
        );
    }
    decoder_comparison()?;
    Ok(())
}

fn argmax(x: &[f32]) -> usize {
    (1..x.len()).fold(0, |best, i| if x[i] > x[best] { i } else { best })
}

fn transpose_matrix(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = values[row * cols + col];
        }
    }
    transposed
}

fn decoder_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let c = Config {
        vocab_size: 12,
        context: 6,
        width: 8,
        heads: 2,
        layers: 1,
        ff_width: 16,
    };
    let base = Decoder::new(c, 41)?;
    let input = [1, 4, 2, 7];
    let targets = [4, 2, 7, 3];
    let reference_logits = base.forward(&input)?;
    let reference_loss = base.loss(&input, &targets)?;
    let span = base
        .parameter_spans()
        .into_iter()
        .find(|s| s.name == "output_weight")
        .ok_or("missing output projection")?;
    let output_weight_d_by_v = &base.parameters()[span.start..span.end];
    let output_rows_v_by_d = transpose_matrix(output_weight_d_by_v, c.width, c.vocab_size);
    for bits in [8, 4] {
        let q = quantize(&output_rows_v_by_d, c.vocab_size, c.width, bits)?;
        let dequantized_d_by_v = transpose_matrix(&q.dequantize(), c.vocab_size, c.width);
        let mut model = base.clone();
        model.parameters_mut()[span.start..span.end].copy_from_slice(&dequantized_d_by_v);
        let logits = model.forward(&input)?;
        let (_, max_logit_error) = quantization_error(&reference_logits, &logits)?;
        let loss = model.loss(&input, &targets)?;
        let token = argmax(&logits[logits.len() - c.vocab_size..]);
        println!("decoder int{bits}: loss={loss:.6} (fp32={reference_loss:.6}), max_logit_error={max_logit_error:.6}, final_argmax={token}");
    }
    Ok(())
}

/// Run the completed algorithm; the browser never executes this Rust.
pub fn run(args: &[String]) -> Result<(), String> {
    if args == ["--bench"] {
        return bench().map_err(|e| e.to_string());
    }
    if !args.is_empty() {
        return Err("unexpected experiment argument".into());
    }
    experiment().map_err(|e| e.to_string())
}

/// Integer-only dot products use a wider accumulator than their i8 operands.
fn integer_dot(left: &[i8], right: &[i8]) -> Result<i32, &'static str> {
    if left.is_empty() || left.len() != right.len() {
        return Err("integer dot requires equal nonempty vectors");
    }
    left.iter().zip(right).try_fold(0_i32, |sum, (&a, &b)| {
        sum.checked_add(i32::from(a) * i32::from(b))
            .ok_or("int32 accumulator overflow")
    })
}
pub fn check() -> Result<(), String> {
    let verify = || -> Result<(), Box<dyn std::error::Error>> {
        let q = quantize(&[-7., -1., 0., 3., 7.], 1, 5, 4)?;
        if q.data.len() != 3 || q.dequantize() != [-7., -1., 0., 3., 7.] {
            return Err(
                "GOAL_NOT_MET: pack signed int4 including odd tails; one byte per code is not int4 storage"
                    .into(),
            );
        }
        let w = fixture(4, 9);
        let x = vec![0.25; 9];
        let reference = dense_matvec_reference(&w, 4, 9, &x);
        let a = quantize(&w, 4, 9, 8)?;
        let b = quantize(&w, 4, 9, 4)?;
        if a.storage_bytes() != 52 || b.storage_bytes() != 34 {
            return Err("GOAL_NOT_MET: 4x9 byte accounting must include four f32 scales".into());
        }
        if quantization_error(&reference, &a.matvec(&x)?)?.0 > 0.02
            || quantization_error(&reference, &b.matvec(&x)?)?.0 > 0.3
        {
            return Err(
                "GOAL_NOT_MET: per-row quantization exceeds held-out output tolerance".into(),
            );
        }
        if integer_dot(&[127; 3], &[127; 3])? != 48_387 {
            return Err("GOAL_NOT_MET: integer dot must accumulate beyond int8 range".into());
        }
        decoder_comparison()?;
        println!("goal: signed packing, per-row scales, wide accumulation and real decoder comparison pass");
        Ok(())
    };
    verify().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packing_round_trips_signed_int4() {
        let q = quantize(&[-7.0, -1.0, 0.0, 3.0, 7.0], 1, 5, 4).unwrap();
        assert_eq!(
            (0..5).map(|i| q.value(i)).collect::<Vec<_>>(),
            [-7, -1, 0, 3, 7]
        );
        assert_eq!(q.data.len(), 3);
    }
    #[test]
    fn output_weight_layout_round_trips_d_by_v() {
        let d_by_v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let v_by_d = transpose_matrix(&d_by_v, 2, 3);
        assert_eq!(v_by_d, [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(transpose_matrix(&v_by_d, 3, 2), d_by_v);
    }
    #[test]
    fn quantized_matvec_is_close_and_int4_is_smaller() {
        let weights = fixture(4, 9);
        let input = vec![0.25; 9];
        let reference = dense_matvec_reference(&weights, 4, 9, &input);
        let q8 = quantize(&weights, 4, 9, 8).unwrap();
        let q4 = quantize(&weights, 4, 9, 4).unwrap();
        assert!(
            quantization_error(&reference, &q8.matvec(&input).unwrap())
                .unwrap()
                .0
                < 0.02
        );
        assert!(
            quantization_error(&reference, &q4.matvec(&input).unwrap())
                .unwrap()
                .0
                < 0.3
        );
        assert!(q4.storage_bytes() < q8.storage_bytes());
    }
    #[test]
    fn rejects_bad_shapes_and_nonfinite_values() {
        assert!(quantize(&[1.0], 1, 2, 8).is_err());
        assert!(quantize(&[f32::NAN], 1, 1, 8).is_err());
        let q = quantize(&[f32::from_bits(1)], 1, 1, 8).unwrap();
        assert!(q.scales[0] > 0.0 && q.matvec(&[1.0]).unwrap()[0].is_finite());
        assert!(quantize(&[f32::MAX], 1, 1, 8)
            .unwrap()
            .matvec(&[2.0])
            .is_err());
        assert!(quantization_error(&[1.0], &[]).is_err());
        assert!(quantization_error(&[1.0], &[f32::NAN]).is_err());
    }

    #[test]
    fn complete_learning_goals() {
        assert!(super::check().is_ok());
    }
}
