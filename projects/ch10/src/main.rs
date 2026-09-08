//! IDX loading and a real softmax-regression training loop, using only std.
use std::{env, fs};
const CLASSES: usize = 10;

#[derive(Clone, Debug)]
struct Dataset {
    images: Vec<f64>,
    labels: Vec<u8>,
    rows: usize,
    cols: usize,
}
impl Dataset {
    fn len(&self) -> usize {
        self.labels.len()
    }
    fn features(&self) -> usize {
        self.rows * self.cols
    }
    fn image(&self, n: usize) -> &[f64] {
        let d = self.features();
        &self.images[n * d..(n + 1) * d]
    }
}

fn be_u32(bytes: &[u8], at: usize) -> Result<usize, String> {
    let s = bytes.get(at..at + 4).ok_or("truncated IDX header")?;
    Ok(u32::from_be_bytes(s.try_into().unwrap()) as usize)
}
fn parse_idx(images: &[u8], labels: &[u8]) -> Result<Dataset, String> {
    if be_u32(images, 0)? != 2051 {
        return Err("image IDX magic must be 2051".into());
    }
    if be_u32(labels, 0)? != 2049 {
        return Err("label IDX magic must be 2049".into());
    }
    let n = be_u32(images, 4)?;
    let nl = be_u32(labels, 4)?;
    let rows = be_u32(images, 8)?;
    let cols = be_u32(images, 12)?;
    if n == 0 || rows == 0 || cols == 0 {
        return Err("IDX dimensions and count must be positive".into());
    }
    if n != nl {
        return Err("image and label counts differ".into());
    }
    let pixels = n
        .checked_mul(rows)
        .and_then(|x| x.checked_mul(cols))
        .ok_or("IDX dimensions overflow")?;
    let image_len = pixels.checked_add(16).ok_or("IDX image length overflow")?;
    let label_len = n.checked_add(8).ok_or("IDX label length overflow")?;
    if images.len() != image_len || labels.len() != label_len {
        return Err("IDX payload length does not match header".into());
    }
    let ys = labels[8..].to_vec();
    if ys.iter().any(|&y| y as usize >= CLASSES) {
        return Err("label is outside 0..9".into());
    }
    Ok(Dataset {
        images: images[16..].iter().map(|&x| x as f64 / 255.0).collect(),
        labels: ys,
        rows,
        cols,
    })
}
fn load_idx(image_path: &str, label_path: &str) -> Result<Dataset, String> {
    let i = fs::read(image_path).map_err(|e| format!("cannot read {image_path}: {e}"))?;
    let l = fs::read(label_path).map_err(|e| format!("cannot read {label_path}: {e}"))?;
    parse_idx(&i, &l)
}

struct Linear {
    features: usize,
    w: Vec<f64>,
    b: [f64; CLASSES],
}
impl Linear {
    fn new(features: usize) -> Self {
        Self {
            features,
            w: vec![0.0; CLASSES * features],
            b: [0.0; CLASSES],
        }
    }
    fn logits(&self, x: &[f64]) -> [f64; CLASSES] {
        let mut z = self.b;
        for (c, output) in z.iter_mut().enumerate() {
            for (j, &v) in x.iter().enumerate() {
                *output += self.w[c * self.features + j] * v
            }
        }
        z
    }
    fn probabilities(logits: [f64; CLASSES]) -> [f64; CLASSES] {
        let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut p = logits.map(|x| (x - m).exp());
        let s = p.iter().sum::<f64>();
        for x in &mut p {
            *x /= s
        }
        p
    }
    fn cross_entropy(logits: [f64; CLASSES], target: usize) -> f64 {
        let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (m - logits[target]) + logits.iter().map(|z| (z - m).exp()).sum::<f64>().ln()
    }
    fn train(
        &mut self,
        data: &Dataset,
        epochs: usize,
        rate: f64,
        batch: usize,
    ) -> Result<(), String> {
        if data.features() != self.features || data.len() == 0 {
            return Err("training dataset shape is incompatible".into());
        }
        if epochs == 0 || batch == 0 || !rate.is_finite() || rate <= 0.0 {
            return Err("epochs, batch, and rate must be positive".into());
        }
        for _ in 0..epochs {
            for start in (0..data.len()).step_by(batch) {
                let end = (start + batch).min(data.len());
                let mut gw = vec![0.0; self.w.len()];
                let mut gb = [0.0; CLASSES];
                for n in start..end {
                    let mut p = Self::probabilities(self.logits(data.image(n)));
                    p[data.labels[n] as usize] -= 1.0;
                    for (c, (&probability, bias_gradient)) in p.iter().zip(&mut gb).enumerate() {
                        *bias_gradient += probability;
                        for (j, &input) in data.image(n).iter().enumerate() {
                            gw[c * self.features + j] += probability * input
                        }
                    }
                }
                let scale = rate / (end - start) as f64;
                for (i, g) in self.w.iter_mut().zip(gw) {
                    *i -= scale * g
                }
                for (bias, gradient) in self.b.iter_mut().zip(gb) {
                    *bias -= scale * gradient
                }
            }
        }
        Ok(())
    }
    fn metrics(&self, data: &Dataset) -> Result<(f64, f64), String> {
        if data.features() != self.features || data.len() == 0 {
            return Err("evaluation dataset shape is incompatible".into());
        }
        let (mut loss, mut correct) = (0.0, 0usize);
        for n in 0..data.len() {
            let z = self.logits(data.image(n));
            loss += Self::cross_entropy(z, data.labels[n] as usize);
            let pred = (0..CLASSES).max_by(|&a, &b| z[a].total_cmp(&z[b])).unwrap();
            correct += (pred == data.labels[n] as usize) as usize;
        }
        Ok((loss / data.len() as f64, correct as f64 / data.len() as f64))
    }
}

fn idx_bytes(samples: &[([u8; 4], u8)]) -> (Vec<u8>, Vec<u8>) {
    let mut i = Vec::new();
    for x in [2051u32, samples.len() as u32, 2, 2] {
        i.extend(x.to_be_bytes())
    }
    let mut l = Vec::new();
    for x in [2049u32, samples.len() as u32] {
        l.extend(x.to_be_bytes())
    }
    for (s, y) in samples {
        i.extend(s);
        l.push(*y)
    }
    (i, l)
}
fn fixture(train: bool) -> Dataset {
    let samples = if train {
        vec![
            ([255, 255, 255, 255], 0),
            ([230, 255, 255, 230], 0),
            ([0, 255, 0, 255], 1),
            ([0, 230, 0, 255], 1),
            ([255, 255, 0, 0], 2),
            ([255, 230, 0, 0], 2),
        ]
    } else {
        vec![
            ([240, 240, 255, 255], 0),
            ([0, 255, 0, 240], 1),
            ([240, 255, 0, 0], 2),
        ]
    };
    let (i, l) = idx_bytes(&samples);
    parse_idx(&i, &l).unwrap()
}

fn run(train: Dataset, held_out: Dataset, epochs: usize, rate: f64) -> Result<(), String> {
    if train.rows != held_out.rows || train.cols != held_out.cols {
        return Err("train and held-out image dimensions differ".into());
    }
    let mut m = Linear::new(train.features());
    let before = m.metrics(&held_out)?;
    m.train(&train, epochs, rate, 32)?;
    let after = m.metrics(&held_out)?;
    println!(
        "train={} held-out={} shape={}x{}",
        train.len(),
        held_out.len(),
        train.rows,
        train.cols
    );
    println!(
        "held-out loss {:.4} -> {:.4}, accuracy {:.1}% -> {:.1}%",
        before.0,
        after.0,
        100. * before.1,
        100. * after.1
    );
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = env::args().collect();
    match a.as_slice() {
        [_] => run(fixture(true), fixture(false), 400, 0.5)?,
        [_, flag, ti, tl, vi, vl] if flag == "--mnist" => {
            run(load_idx(ti, tl)?, load_idx(vi, vl)?, 1, 0.1)?
        }
        [_, flag, ti, tl, vi, vl, e] if flag == "--mnist" => run(
            load_idx(ti, tl)?,
            load_idx(vi, vl)?,
            e.parse().map_err(|_| "epochs must be an integer")?,
            0.1,
        )?,
        _ => {
            return Err(
                "usage: ch10 [--mnist TRAIN_IMAGES TRAIN_LABELS TEST_IMAGES TEST_LABELS [EPOCHS]]"
                    .into(),
            )
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_real_idx_layout() {
        let d = fixture(true);
        assert_eq!((d.len(), d.rows, d.cols), (6, 2, 2));
        assert_eq!(d.labels, [0, 0, 1, 1, 2, 2]);
    }
    #[test]
    fn malformed_idx_is_rejected() {
        let (mut i, l) = idx_bytes(&[([0; 4], 0)]);
        i[3] = 0;
        assert!(parse_idx(&i, &l).unwrap_err().contains("magic"));
    }
    #[test]
    fn training_reduces_real_cross_entropy() {
        let tr = fixture(true);
        let te = fixture(false);
        let mut m = Linear::new(4);
        let before = m.metrics(&te).unwrap().0;
        m.train(&tr, 400, 0.5, 3).unwrap();
        let (loss, acc) = m.metrics(&te).unwrap();
        assert!(loss < before * 0.2 && acc == 1.0);
    }
    #[test]
    fn cross_entropy_preserves_shared_offsets_and_parser_contract() {
        assert!((Linear::cross_entropy([1e16; CLASSES], 0) - (CLASSES as f64).ln()).abs() < 1e-12);
        let (images, labels) = idx_bytes(&[([0; 4], 0)]);
        assert!(parse_idx(&images[..images.len() - 1], &labels).is_err());
        let mut bad = labels.clone();
        bad[8] = 10;
        assert!(parse_idx(&images, &bad).is_err());
        bad[4..8].copy_from_slice(&2u32.to_be_bytes());
        assert!(parse_idx(&images, &bad).is_err());
    }
}
