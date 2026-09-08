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
    fn in_features(&self) -> usize {
        self.rows * self.cols
    }
    fn image(&self, n: usize) -> &[f64] {
        let in_features = self.in_features();
        &self.images[n * in_features..(n + 1) * in_features]
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
    let image_count = be_u32(images, 4)?;
    let label_count = be_u32(labels, 4)?;
    let rows = be_u32(images, 8)?;
    let cols = be_u32(images, 12)?;
    if image_count == 0 || rows == 0 || cols == 0 {
        return Err("IDX dimensions and count must be positive".into());
    }
    if image_count != label_count {
        return Err("image and label counts differ".into());
    }
    let pixels = image_count
        .checked_mul(rows)
        .and_then(|x| x.checked_mul(cols))
        .ok_or("IDX dimensions overflow")?;
    let image_len = pixels.checked_add(16).ok_or("IDX image length overflow")?;
    let label_len = image_count
        .checked_add(8)
        .ok_or("IDX label length overflow")?;
    if images.len() != image_len || labels.len() != label_len {
        return Err("IDX payload length does not match header".into());
    }
    let parsed_labels = labels[8..].to_vec();
    if parsed_labels.iter().any(|&label| label as usize >= CLASSES) {
        return Err("label is outside 0..9".into());
    }
    Ok(Dataset {
        images: images[16..].iter().map(|&x| x as f64 / 255.0).collect(),
        labels: parsed_labels,
        rows,
        cols,
    })
}
fn load_idx(image_path: &str, label_path: &str) -> Result<Dataset, String> {
    let image_bytes = fs::read(image_path).map_err(|e| format!("cannot read {image_path}: {e}"))?;
    let label_bytes = fs::read(label_path).map_err(|e| format!("cannot read {label_path}: {e}"))?;
    parse_idx(&image_bytes, &label_bytes)
}

struct Linear {
    in_features: usize,
    weights: Vec<f64>,
    bias: [f64; CLASSES],
}
impl Linear {
    fn new(in_features: usize) -> Self {
        Self {
            in_features,
            weights: vec![0.0; CLASSES * in_features],
            bias: [0.0; CLASSES],
        }
    }
    fn logits(&self, features: &[f64]) -> [f64; CLASSES] {
        let mut logits = self.bias;
        for (class, logit) in logits.iter_mut().enumerate() {
            for (feature, &input) in features.iter().enumerate() {
                *logit += self.weights[class * self.in_features + feature] * input
            }
        }
        logits
    }
    fn probabilities(logits: [f64; CLASSES]) -> [f64; CLASSES] {
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut probabilities = logits.map(|logit| (logit - maximum).exp());
        let sum = probabilities.iter().sum::<f64>();
        for probability in &mut probabilities {
            *probability /= sum
        }
        probabilities
    }
    fn cross_entropy_from_logits(logits: [f64; CLASSES], target: usize) -> f64 {
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (maximum - logits[target])
            + logits
                .iter()
                .map(|logit| (logit - maximum).exp())
                .sum::<f64>()
                .ln()
    }
    fn train(
        &mut self,
        data: &Dataset,
        epochs: usize,
        learning_rate: f64,
        batch_size: usize,
    ) -> Result<(), String> {
        if data.in_features() != self.in_features || data.len() == 0 {
            return Err("training dataset shape is incompatible".into());
        }
        if epochs == 0 || batch_size == 0 || !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("epochs, batch size, and learning rate must be positive".into());
        }
        for _ in 0..epochs {
            for start in (0..data.len()).step_by(batch_size) {
                let end = (start + batch_size).min(data.len());
                let mut weight_gradients = vec![0.0; self.weights.len()];
                let mut bias_gradients = [0.0; CLASSES];
                for n in start..end {
                    let mut logit_gradients = Self::probabilities(self.logits(data.image(n)));
                    logit_gradients[data.labels[n] as usize] -= 1.0;
                    for (class, (&logit_gradient, bias_gradient)) in
                        logit_gradients.iter().zip(&mut bias_gradients).enumerate()
                    {
                        *bias_gradient += logit_gradient;
                        for (feature, &input) in data.image(n).iter().enumerate() {
                            weight_gradients[class * self.in_features + feature] +=
                                logit_gradient * input
                        }
                    }
                }
                let scale = learning_rate / (end - start) as f64;
                for (weight, gradient) in self.weights.iter_mut().zip(weight_gradients) {
                    *weight -= scale * gradient
                }
                for (bias, gradient) in self.bias.iter_mut().zip(bias_gradients) {
                    *bias -= scale * gradient
                }
            }
        }
        Ok(())
    }
    fn metrics(&self, data: &Dataset) -> Result<(f64, f64), String> {
        if data.in_features() != self.in_features || data.len() == 0 {
            return Err("evaluation dataset shape is incompatible".into());
        }
        let (mut loss, mut correct) = (0.0, 0usize);
        for n in 0..data.len() {
            let logits = self.logits(data.image(n));
            let target = data.labels[n] as usize;
            loss += Self::cross_entropy_from_logits(logits, target);
            let prediction = (0..CLASSES)
                .max_by(|&left, &right| logits[left].total_cmp(&logits[right]))
                .unwrap();
            correct += (prediction == target) as usize;
        }
        Ok((loss / data.len() as f64, correct as f64 / data.len() as f64))
    }
}

fn idx_bytes(samples: &[([u8; 4], u8)]) -> (Vec<u8>, Vec<u8>) {
    let mut images = Vec::new();
    for value in [2051u32, samples.len() as u32, 2, 2] {
        images.extend(value.to_be_bytes())
    }
    let mut labels = Vec::new();
    for value in [2049u32, samples.len() as u32] {
        labels.extend(value.to_be_bytes())
    }
    for (pixels, label) in samples {
        images.extend(pixels);
        labels.push(*label)
    }
    (images, labels)
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
    let (images, labels) = idx_bytes(&samples);
    parse_idx(&images, &labels).unwrap()
}

fn run(
    train_data: Dataset,
    validation_data: Dataset,
    epochs: usize,
    learning_rate: f64,
) -> Result<(), String> {
    if train_data.rows != validation_data.rows || train_data.cols != validation_data.cols {
        return Err("train and held-out image dimensions differ".into());
    }
    let mut model = Linear::new(train_data.in_features());
    let before = model.metrics(&validation_data)?;
    model.train(&train_data, epochs, learning_rate, 32)?;
    let after = model.metrics(&validation_data)?;
    println!(
        "train={} held-out={} shape={}x{}",
        train_data.len(),
        validation_data.len(),
        train_data.rows,
        train_data.cols
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
    let args: Vec<String> = env::args().collect();
    match args.as_slice() {
        [_] => run(fixture(true), fixture(false), 400, 0.5)?,
        [_, flag, train_images, train_labels, validation_images, validation_labels]
            if flag == "--mnist" =>
        {
            run(
                load_idx(train_images, train_labels)?,
                load_idx(validation_images, validation_labels)?,
                1,
                0.1,
            )?
        }
        [_, flag, train_images, train_labels, validation_images, validation_labels, epochs]
            if flag == "--mnist" =>
        {
            run(
                load_idx(train_images, train_labels)?,
                load_idx(validation_images, validation_labels)?,
                epochs.parse().map_err(|_| "epochs must be an integer")?,
                0.1,
            )?
        }
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
        let (mut images, labels) = idx_bytes(&[([0; 4], 0)]);
        images[3] = 0;
        assert!(parse_idx(&images, &labels).unwrap_err().contains("magic"));
    }
    #[test]
    fn training_reduces_real_cross_entropy() {
        let train_data = fixture(true);
        let validation_data = fixture(false);
        let mut model = Linear::new(4);
        let before = model.metrics(&validation_data).unwrap().0;
        model.train(&train_data, 400, 0.5, 3).unwrap();
        let (loss, accuracy) = model.metrics(&validation_data).unwrap();
        assert!(loss < before * 0.2 && accuracy == 1.0);
    }
    #[test]
    fn cross_entropy_preserves_shared_offsets_and_parser_contract() {
        assert!(
            (Linear::cross_entropy_from_logits([1e16; CLASSES], 0) - (CLASSES as f64).ln()).abs()
                < 1e-12
        );
        let (images, labels) = idx_bytes(&[([0; 4], 0)]);
        assert!(parse_idx(&images[..images.len() - 1], &labels).is_err());
        let mut bad = labels.clone();
        bad[8] = 10;
        assert!(parse_idx(&images, &bad).is_err());
        bad[4..8].copy_from_slice(&2u32.to_be_bytes());
        assert!(parse_idx(&images, &bad).is_err());
    }
}
