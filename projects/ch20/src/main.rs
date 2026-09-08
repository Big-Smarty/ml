//! A two-filter convolutional digit classifier with full backpropagation.
use std::{env, fs};

const CLASSES: usize = 10;
const FILTERS: usize = 2;

#[derive(Clone)]
struct Dataset {
    images: Vec<f64>,
    labels: Vec<usize>,
    rows: usize,
    cols: usize,
}

impl Dataset {
    fn len(&self) -> usize {
        self.labels.len()
    }
    fn image(&self, n: usize) -> &[f64] {
        let d = self.rows * self.cols;
        &self.images[n * d..(n + 1) * d]
    }
}

fn be_u32(bytes: &[u8], at: usize) -> Result<usize, String> {
    let word = bytes.get(at..at + 4).ok_or("truncated IDX header")?;
    Ok(u32::from_be_bytes(word.try_into().unwrap()) as usize)
}

fn parse_idx(images: &[u8], labels: &[u8]) -> Result<Dataset, String> {
    if be_u32(images, 0)? != 2051 || be_u32(labels, 0)? != 2049 {
        return Err("expected IDX image magic 2051 and label magic 2049".into());
    }
    let n = be_u32(images, 4)?;
    let rows = be_u32(images, 8)?;
    let cols = be_u32(images, 12)?;
    if n == 0 || rows == 0 || cols == 0 || be_u32(labels, 4)? != n {
        return Err("IDX counts and dimensions must be positive and agree".into());
    }
    let pixels = n
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(cols))
        .ok_or("IDX dimensions overflow")?;
    let image_len = 16usize
        .checked_add(pixels)
        .ok_or("IDX image length overflow")?;
    let label_len = 8usize.checked_add(n).ok_or("IDX label length overflow")?;
    if images.len() != image_len || labels.len() != label_len {
        return Err("IDX payload length does not match its header".into());
    }
    if labels[8..].iter().any(|&y| y as usize >= CLASSES) {
        return Err("MNIST label must be in 0..9".into());
    }
    Ok(Dataset {
        images: images[16..].iter().map(|&v| v as f64 / 255.0).collect(),
        labels: labels[8..].iter().map(|&v| v as usize).collect(),
        rows,
        cols,
    })
}

fn load_idx(image_path: &str, label_path: &str) -> Result<Dataset, String> {
    let images = fs::read(image_path).map_err(|e| format!("cannot read {image_path}: {e}"))?;
    let labels = fs::read(label_path).map_err(|e| format!("cannot read {label_path}: {e}"))?;
    parse_idx(&images, &labels)
}

#[derive(Clone)]
struct Model {
    rows: usize,
    cols: usize,
    kernel: [[f64; 9]; FILTERS],
    conv_bias: [f64; FILTERS],
    dense: Vec<f64>,
    class_bias: [f64; CLASSES],
}

struct Forward {
    relu: Vec<f64>,
    winners: Vec<usize>,
    pooled: Vec<f64>,
    logits: [f64; CLASSES],
}

struct Grad {
    kernel: [[f64; 9]; FILTERS],
    conv_bias: [f64; FILTERS],
    dense: Vec<f64>,
    class_bias: [f64; CLASSES],
}

impl Model {
    fn new(rows: usize, cols: usize) -> Result<Self, String> {
        if rows < 4 || cols < 4 {
            return Err("images must be at least 4x4".into());
        }
        let pooled = FILTERS
            .checked_mul(rows / 2)
            .and_then(|v| v.checked_mul(cols / 2))
            .ok_or("model shape overflows addressable memory")?;
        let dense_len = CLASSES
            .checked_mul(pooled)
            .ok_or("dense shape overflows addressable memory")?;
        let mut dense = vec![0.0; dense_len];
        for (i, value) in dense.iter_mut().enumerate() {
            *value = ((i * 37 + 11) as f64).sin() * 0.08;
        }
        Ok(Self {
            rows,
            cols,
            kernel: [
                [-0.08, 0.02, 0.08, -0.12, 0.03, 0.12, -0.08, 0.02, 0.08],
                [-0.08, -0.12, -0.08, 0.02, 0.03, 0.02, 0.08, 0.12, 0.08],
            ],
            conv_bias: [0.02; FILTERS],
            dense,
            class_bias: [0.0; CLASSES],
        })
    }

    fn forward(&self, image: &[f64]) -> Forward {
        assert_eq!(image.len(), self.rows * self.cols);
        let plane = self.rows * self.cols;
        let mut relu = vec![0.0; FILTERS * plane];
        for f in 0..FILTERS {
            for r in 0..self.rows {
                for c in 0..self.cols {
                    let mut z = self.conv_bias[f];
                    for kr in 0..3 {
                        for kc in 0..3 {
                            let ir = r as isize + kr as isize - 1;
                            let ic = c as isize + kc as isize - 1;
                            if ir >= 0
                                && ic >= 0
                                && ir < self.rows as isize
                                && ic < self.cols as isize
                            {
                                z += self.kernel[f][kr * 3 + kc]
                                    * image[ir as usize * self.cols + ic as usize];
                            }
                        }
                    }
                    relu[f * plane + r * self.cols + c] = z.max(0.0);
                }
            }
        }
        let pr = self.rows / 2;
        let pc = self.cols / 2;
        let mut pooled = Vec::with_capacity(FILTERS * pr * pc);
        let mut winners = Vec::with_capacity(FILTERS * pr * pc);
        for f in 0..FILTERS {
            for r in 0..pr {
                for c in 0..pc {
                    let indices = [
                        f * plane + (2 * r) * self.cols + 2 * c,
                        f * plane + (2 * r) * self.cols + 2 * c + 1,
                        f * plane + (2 * r + 1) * self.cols + 2 * c,
                        f * plane + (2 * r + 1) * self.cols + 2 * c + 1,
                    ];
                    let winner = *indices
                        .iter()
                        .max_by(|&&a, &&b| relu[a].total_cmp(&relu[b]))
                        .unwrap();
                    pooled.push(relu[winner]);
                    winners.push(winner);
                }
            }
        }
        let mut logits = self.class_bias;
        for (y, logit) in logits.iter_mut().enumerate() {
            for (j, &value) in pooled.iter().enumerate() {
                *logit += self.dense[y * pooled.len() + j] * value;
            }
        }
        Forward {
            relu,
            winners,
            pooled,
            logits,
        }
    }

    fn loss_and_grad(&self, image: &[f64], label: usize) -> (f64, Grad) {
        let pass = self.forward(image);
        let max = pass
            .logits
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = pass.logits.iter().map(|&z| (z - max).exp()).sum();
        let loss = (max - pass.logits[label]) + sum_exp.ln();
        let mut dz = pass.logits.map(|z| (z - max).exp() / sum_exp);
        dz[label] -= 1.0;
        let mut grad = Grad {
            kernel: [[0.0; 9]; FILTERS],
            conv_bias: [0.0; FILTERS],
            dense: vec![0.0; self.dense.len()],
            class_bias: dz,
        };
        let mut dpool = vec![0.0; pass.pooled.len()];
        for (y, &dy) in dz.iter().enumerate() {
            for (j, dpool_value) in dpool.iter_mut().enumerate() {
                grad.dense[y * pass.pooled.len() + j] = dy * pass.pooled[j];
                *dpool_value += self.dense[y * pass.pooled.len() + j] * dy;
            }
        }
        let plane = self.rows * self.cols;
        let mut drelu = vec![0.0; pass.relu.len()];
        for (j, &winner) in pass.winners.iter().enumerate() {
            drelu[winner] += dpool[j];
        }
        for f in 0..FILTERS {
            for r in 0..self.rows {
                for c in 0..self.cols {
                    let at = f * plane + r * self.cols + c;
                    if pass.relu[at] <= 0.0 {
                        continue;
                    }
                    let d = drelu[at];
                    grad.conv_bias[f] += d;
                    for kr in 0..3 {
                        for kc in 0..3 {
                            let ir = r as isize + kr as isize - 1;
                            let ic = c as isize + kc as isize - 1;
                            if ir >= 0
                                && ic >= 0
                                && ir < self.rows as isize
                                && ic < self.cols as isize
                            {
                                grad.kernel[f][kr * 3 + kc] +=
                                    d * image[ir as usize * self.cols + ic as usize];
                            }
                        }
                    }
                }
            }
        }
        (loss, grad)
    }

    fn train(
        &mut self,
        data: &Dataset,
        epochs: usize,
        rate: f64,
        limit: usize,
    ) -> Result<(), String> {
        if data.rows != self.rows || data.cols != self.cols || data.len() == 0 {
            return Err("training data shape does not match the model".into());
        }
        if epochs == 0 || !rate.is_finite() || rate <= 0.0 {
            return Err("epochs and learning rate must be positive".into());
        }
        let n = data.len().min(limit);
        if n == 0 {
            return Err("training limit must include at least one row".into());
        }
        for _ in 0..epochs {
            for i in 0..n {
                let (loss, g) = self.loss_and_grad(data.image(i), data.labels[i]);
                if !loss.is_finite() {
                    return Err("non-finite training loss; reduce the learning rate".into());
                }
                for f in 0..FILTERS {
                    for k in 0..9 {
                        self.kernel[f][k] -= rate * g.kernel[f][k];
                    }
                    self.conv_bias[f] -= rate * g.conv_bias[f];
                }
                for (w, dw) in self.dense.iter_mut().zip(g.dense) {
                    *w -= rate * dw;
                }
                for y in 0..CLASSES {
                    self.class_bias[y] -= rate * g.class_bias[y];
                }
                if self
                    .kernel
                    .iter()
                    .flatten()
                    .chain(&self.conv_bias)
                    .chain(&self.dense)
                    .chain(&self.class_bias)
                    .any(|x| !x.is_finite())
                {
                    return Err("training update overflow; reduce the learning rate".into());
                }
            }
        }
        Ok(())
    }

    fn metrics(&self, data: &Dataset, limit: usize) -> Result<(f64, f64), String> {
        if data.rows != self.rows || data.cols != self.cols || data.len() == 0 {
            return Err("evaluation data shape does not match the model".into());
        }
        let n = data.len().min(limit);
        if n == 0 {
            return Err("evaluation limit must include at least one row".into());
        }
        let mut loss = 0.0;
        let mut correct = 0;
        for i in 0..n {
            let pass = self.forward(data.image(i));
            let max = pass
                .logits
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            loss += (max - pass.logits[data.labels[i]])
                + pass
                    .logits
                    .iter()
                    .map(|&z| (z - max).exp())
                    .sum::<f64>()
                    .ln();
            let guess = (0..CLASSES)
                .max_by(|&a, &b| pass.logits[a].total_cmp(&pass.logits[b]))
                .unwrap();
            correct += usize::from(guess == data.labels[i]);
        }
        if !loss.is_finite() {
            return Err("non-finite evaluation loss".into());
        }
        Ok((loss / n as f64, correct as f64 / n as f64))
    }
}

fn seven_segment(digit: usize, shift: isize, noise: usize) -> Vec<f64> {
    const MAP: [[bool; 7]; 10] = [
        [true, true, true, false, true, true, true],
        [false, false, true, false, false, true, false],
        [true, false, true, true, true, false, true],
        [true, false, true, true, false, true, true],
        [false, true, true, true, false, true, false],
        [true, true, false, true, false, true, true],
        [true, true, false, true, true, true, true],
        [true, false, true, false, false, true, false],
        [true, true, true, true, true, true, true],
        [true, true, true, true, false, true, true],
    ];
    let mut image = vec![0.0; 64];
    let segments = [
        (1, 1, 2, 5),
        (1, 3, 1, 1),
        (1, 3, 6, 6),
        (3, 3, 2, 5),
        (3, 5, 1, 1),
        (3, 5, 6, 6),
        (5, 5, 2, 5),
    ];
    for (on, &(r0, r1, c0, c1)) in MAP[digit].iter().zip(&segments) {
        if *on {
            for r in r0..=r1 {
                for c in c0..=c1 {
                    let cc = c as isize + shift;
                    if (0..8).contains(&cc) {
                        image[r * 8 + cc as usize] = 1.0;
                    }
                }
            }
        }
    }
    image[(digit * 11 + noise * 17) % 64] *= 0.75;
    image
}

fn fixture(train: bool) -> Dataset {
    let variants = if train { &[-1, 0, 1][..] } else { &[0][..] };
    let mut images = Vec::new();
    let mut labels = Vec::new();
    for digit in 0..10 {
        for (noise, &shift) in variants.iter().enumerate() {
            let mut image = seven_segment(digit, shift, noise);
            if !train {
                for value in &mut image {
                    *value *= 0.9;
                }
                image[(digit * 13 + 7) % 64] += 0.05;
            }
            images.extend(image);
            labels.push(digit);
        }
    }
    Dataset {
        images,
        labels,
        rows: 8,
        cols: 8,
    }
}

fn run(
    train: Dataset,
    valid: Dataset,
    epochs: usize,
    rate: f64,
    limit: usize,
) -> Result<(), String> {
    if train.rows != valid.rows || train.cols != valid.cols {
        return Err("fit and validation image dimensions differ".into());
    }
    let mut model = Model::new(train.rows, train.cols)?;
    let before = model.metrics(&valid, limit)?;
    model.train(&train, epochs, rate, limit)?;
    let after = model.metrics(&valid, limit)?;
    println!(
        "fit={} validation={} shape={}x{}",
        train.len().min(limit),
        valid.len().min(limit),
        train.rows,
        train.cols
    );
    println!(
        "validation cross-entropy {:.4} -> {:.4}; accuracy {:.1}% -> {:.1}%",
        before.0,
        after.0,
        before.1 * 100.0,
        after.1 * 100.0
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    match args.as_slice() {
        [_] => run(fixture(true), fixture(false), 80, 0.025, usize::MAX)?,
        [_, flag, fi, fl, vi, vl] if flag == "--mnist" => {
            run(load_idx(fi, fl)?, load_idx(vi, vl)?, 1, 0.01, 2_000)?
        }
        _ => {
            return Err(
                "usage: ch20 [--mnist FIT_IMAGES FIT_LABELS VALID_IMAGES VALID_LABELS]".into(),
            )
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cross_entropy_preserves_small_loss_at_large_common_offset() {
        let mut model = Model::new(8, 8).unwrap();
        model.dense.fill(0.0);
        model.class_bias.fill(1e16);
        let data = fixture(false);
        let expected = (CLASSES as f64).ln();
        assert!((model.loss_and_grad(data.image(0), data.labels[0]).0 - expected).abs() < 1e-12);
        assert!((model.metrics(&data, 1).unwrap().0 - expected).abs() < 1e-12);
    }
    #[test]
    fn idx_payload_is_decoded_and_validated() {
        let mut images = Vec::new();
        for word in [2051u32, 1, 4, 4] {
            images.extend(word.to_be_bytes());
        }
        images.extend([255; 16]);
        let mut labels = Vec::new();
        for word in [2049u32, 1] {
            labels.extend(word.to_be_bytes());
        }
        labels.push(3);
        let data = parse_idx(&images, &labels).unwrap();
        assert_eq!(data.image(0), &[1.0; 16]);
        assert_eq!(data.labels, vec![3]);
        labels[8] = 10;
        assert!(parse_idx(&images, &labels).is_err());
        labels[8] = 3;
        images.pop();
        assert!(parse_idx(&images, &labels).is_err());
    }

    #[test]
    fn convolution_gradient_matches_central_difference_away_from_ties() {
        let mut model = Model::new(8, 8).unwrap();
        let mut x: Vec<f64> = (0..64).map(|i| (i as f64 * 0.071).sin() + 0.3).collect();
        for (i, value) in x.iter_mut().enumerate() {
            *value += i as f64 * 1e-4;
        }
        let analytic = model.loss_and_grad(&x, 3).1.kernel[0][0];
        let h = 1e-5;
        model.kernel[0][0] += h;
        let plus = model.loss_and_grad(&x, 3).0;
        model.kernel[0][0] -= 2.0 * h;
        let minus = model.loss_and_grad(&x, 3).0;
        let numeric = (plus - minus) / (2.0 * h);
        assert!((analytic - numeric).abs() < 1e-6 + 1e-4 * numeric.abs());
    }

    #[test]
    fn training_reduces_held_out_loss() {
        let train = fixture(true);
        let valid = fixture(false);
        let mut model = Model::new(8, 8).unwrap();
        let before = model.metrics(&valid, usize::MAX).unwrap().0;
        model.train(&train, 80, 0.025, usize::MAX).unwrap();
        let after = model.metrics(&valid, usize::MAX).unwrap();
        assert!(
            after.0 < before * 0.7 && after.1 >= 0.7,
            "{before:?} {after:?}"
        );
    }

    #[test]
    fn malformed_idx_is_rejected() {
        assert!(parse_idx(&[0; 16], &[0; 8]).is_err());
    }

    #[test]
    fn fixture_validation_images_are_disjoint() {
        let train = fixture(true);
        let valid = fixture(false);
        assert!(valid
            .images
            .chunks(64)
            .all(|v| !train.images.chunks(64).any(|t| t == v)));
        assert!(Model::new(usize::MAX, usize::MAX).is_err());
        assert!(Model::new(8, 8).unwrap().metrics(&valid, 0).is_err());
    }
}
