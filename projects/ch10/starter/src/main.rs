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

#[cfg(test)]
fn probabilities_from_logits(logits: &[f64]) -> Vec<f64> {
    let _ = logits;
    // TODO: subtract the maximum, exponentiate, normalize.
    todo!("implement stable softmax")
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    let data = match args.as_slice() {
        [_] => fixture(true),
        [_, images, labels] => load_idx(images, labels)?,
        _ => return Err("usage: ch10-starter [IMAGES_IDX LABELS_IDX]".into()),
    };
    let mut counts = [0usize; CLASSES];
    for &label in &data.labels {
        counts[label as usize] += 1;
    }
    let mean_intensity = data.images.iter().sum::<f64>() / data.images.len() as f64;
    println!(
        "loaded {} images, shape {}x{}, class counts {:?}",
        data.len(),
        data.rows,
        data.cols,
        counts
    );
    println!(
        "mean pixel intensity {mean_intensity:.4}; first image {:?}",
        &data.image(0)[..data.in_features().min(8)]
    );
    println!("Data loading is ready. Complete probabilities_from_logits, then extend the dense layer into ten class logits.");
    Ok(())
}
#[test]
fn probabilities_are_stable_and_normalized() {
    let probabilities = probabilities_from_logits(&[1000.0, 1001.0]);
    assert_eq!(probabilities.len(), 2);
    assert!(probabilities.iter().all(|value| value.is_finite()));
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!((probabilities[0] - 0.2689414213699951).abs() < 1e-12);
    assert_eq!(
        probabilities_from_logits(&[0.0, 0.0, 0.0, 0.0]),
        vec![0.25; 4]
    );
}
