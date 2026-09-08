//! Symmetric per-output-row int8 and packed-int4 weight quantization.
use ch36::{Config, Decoder};
use std::{env, hint::black_box, time::Instant};

#[derive(Debug, Clone)]
struct Quantized {
    rows: usize,
    cols: usize,
    bits: u8,
    scales: Vec<f32>,
    data: Vec<u8>,
}

fn quantize(
    weights: &[f32],
    rows: usize,
    cols: usize,
    bits: u8,
) -> Result<Quantized, &'static str> {
    if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(weights.len()) {
        return Err("weights must be a nonempty [rows, cols] matrix");
    }
    if bits != 8 && bits != 4 {
        return Err("only int8 and int4 are supported");
    }
    if weights.iter().any(|x| !x.is_finite()) {
        return Err("weights must be finite");
    }
    let qmax = if bits == 8 { 127.0 } else { 7.0 };
    let mut scales = Vec::with_capacity(rows);
    let mut values = Vec::with_capacity(weights.len());
    for row in weights.chunks_exact(cols) {
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
            .map(|pair| {
                let low = (pair[0] as u8) & 0x0f;
                let high = pair.get(1).copied().unwrap_or(0) as u8 & 0x0f;
                low | (high << 4)
            })
            .collect()
    };
    Ok(Quantized {
        rows,
        cols,
        bits,
        scales,
        data,
    })
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
        if input.len() != self.cols || input.iter().any(|x| !x.is_finite()) {
            return Err("input must be finite and match the input width");
        }
        let output: Vec<f32> = (0..self.rows)
            .map(|r| {
                (0..self.cols)
                    .map(|c| self.value(r * self.cols + c) as f32 * self.scales[r] * input[c])
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
        (0..self.rows * self.cols)
            .map(|i| self.value(i) as f32 * self.scales[i / self.cols])
            .collect()
    }
}

fn dense_matvec(weights: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|r| (0..cols).map(|c| weights[r * cols + c] * input[c]).sum())
        .collect()
}

fn error(reference: &[f32], candidate: &[f32]) -> Result<(f32, f32), &'static str> {
    if reference.is_empty()
        || reference.len() != candidate.len()
        || reference.iter().chain(candidate).any(|x| !x.is_finite())
    {
        return Err("error comparison needs equal, nonempty, finite slices");
    }
    let mut sum = 0.0;
    let mut max = 0.0_f32;
    for (a, b) in reference.iter().zip(candidate) {
        let e = (a - b).abs();
        sum += e * e;
        max = max.max(e);
    }
    Ok(((sum / reference.len() as f32).sqrt(), max))
}

fn fixture(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0)
        .collect()
}

fn bench() -> Result<(), &'static str> {
    let (rows, cols, repeats) = (256, 256, 200);
    let weights = fixture(rows, cols);
    let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.17).sin()).collect();
    let dense = dense_matvec(&weights, rows, cols, &input);
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
        (samples[3], checksum)
    };
    let (dense_t, dense_sum) =
        measure(&mut || dense_matvec(black_box(&weights), rows, cols, black_box(&input)));
    println!("fp32 median={dense_t:?} for {repeats} matvecs; checksum={dense_sum:.3}");
    for bits in [8, 4] {
        let q = quantize(&weights, rows, cols, bits)?;
        let output = q.matvec(&input)?;
        let (rmse, max) = error(&dense, &output)?;
        let (time, sum) = measure(&mut || q.matvec(black_box(&input)).unwrap());
        println!("int{bits} median={time:?}; rmse={rmse:.5}, max={max:.5}, checksum={sum:.3}");
    }
    println!(
        "Times are local measurements of this dequantizing scalar kernel, not universal speedups."
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match env::args().nth(1).as_deref() {
        Some("--bench") => return bench().map_err(Into::into),
        Some(_) => return Err("usage: ch41 [--bench]".into()),
        None => {}
    }
    let (rows, cols) = (3, 6);
    let weights = fixture(rows, cols);
    let input = [0.5, -1.0, 0.25, 2.0, -0.5, 1.5];
    let dense = dense_matvec(&weights, rows, cols, &input);
    println!(
        "dense output: {dense:?}; fp32 weight bytes={}",
        weights.len() * 4
    );
    for bits in [8, 4] {
        let q = quantize(&weights, rows, cols, bits)?;
        let output = q.matvec(&input)?;
        let (rmse, max) = error(&dense, &output)?;
        println!(
            "int{bits}: output={output:?}, bytes={}, rmse={rmse:.6}, max_abs={max:.6}",
            q.storage_bytes()
        );
    }
    decoder_comparison()?;
    Ok(())
}

fn argmax(x: &[f32]) -> usize {
    (1..x.len()).fold(0, |best, i| if x[i] > x[best] { i } else { best })
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
    let dense_logits = base.forward(&input)?;
    let dense_loss = base.loss_and_grad(&input, &targets)?.loss;
    let span = base
        .parameter_spans()
        .into_iter()
        .find(|s| s.name == "output_weight")
        .ok_or("missing output projection")?;
    let original = &base.parameters()[span.start..span.end];
    let mut rows = vec![0.0; c.vocab_size * c.width];
    for d in 0..c.width {
        for v in 0..c.vocab_size {
            rows[v * c.width + d] = original[d * c.vocab_size + v];
        }
    }
    for bits in [8, 4] {
        let q = quantize(&rows, c.vocab_size, c.width, bits)?;
        let dq = q.dequantize();
        let mut model = base.clone();
        for d in 0..c.width {
            for v in 0..c.vocab_size {
                model.parameters_mut()[span.start + d * c.vocab_size + v] = dq[v * c.width + d];
            }
        }
        let logits = model.forward(&input)?;
        let (_, max) = error(&dense_logits, &logits)?;
        let loss = model.loss_and_grad(&input, &targets)?.loss;
        let token = argmax(&logits[logits.len() - c.vocab_size..]);
        println!("decoder int{bits}: loss={loss:.6} (fp32={dense_loss:.6}), max_logit_error={max:.6}, final_argmax={token}");
    }
    Ok(())
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
    fn quantized_matvec_is_close_and_int4_is_smaller() {
        let w = fixture(4, 9);
        let x = vec![0.25; 9];
        let dense = dense_matvec(&w, 4, 9, &x);
        let q8 = quantize(&w, 4, 9, 8).unwrap();
        let q4 = quantize(&w, 4, 9, 4).unwrap();
        assert!(error(&dense, &q8.matvec(&x).unwrap()).unwrap().0 < 0.02);
        assert!(error(&dense, &q4.matvec(&x).unwrap()).unwrap().0 < 0.3);
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
        assert!(error(&[1.0], &[]).is_err());
        assert!(error(&[1.0], &[f32::NAN]).is_err());
    }
}
