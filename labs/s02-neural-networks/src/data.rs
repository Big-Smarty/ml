//! Supplied IDX plumbing. Parsing is shared by both digit classifiers.
use std::fs;
pub const CLASSES: usize = 10;

#[derive(Clone, Debug)]
pub struct Dataset {
    pub images: Vec<f64>,
    pub labels: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}
impl Dataset {
    pub fn len(&self) -> usize {
        self.labels.len()
    }
    pub fn in_features(&self) -> usize {
        self.rows * self.cols
    }
    pub fn image(&self, n: usize) -> &[f64] {
        let in_features = self.in_features();
        &self.images[n * in_features..(n + 1) * in_features]
    }
}

pub fn be_u32(bytes: &[u8], at: usize) -> Result<usize, String> {
    let s = bytes.get(at..at + 4).ok_or("truncated IDX header")?;
    Ok(u32::from_be_bytes(s.try_into().map_err(|_| "invalid four-byte header")?) as usize)
}
pub fn parse_idx(images: &[u8], labels: &[u8]) -> Result<Dataset, String> {
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
pub fn load_idx(image_path: &str, label_path: &str) -> Result<Dataset, String> {
    let image_bytes = fs::read(image_path).map_err(|e| format!("cannot read {image_path}: {e}"))?;
    let label_bytes = fs::read(label_path).map_err(|e| format!("cannot read {label_path}: {e}"))?;
    parse_idx(&image_bytes, &label_bytes)
}

// Course-authored 5×3 block digits, deliberately not sampled from MNIST.
const GLYPHS: [&str; 10] = [
    "111101101101111",
    "010110010010111",
    "111001111100111",
    "111001111001111",
    "101101111001001",
    "111100111001111",
    "111100111101111",
    "111001010010010",
    "111101111101111",
    "111101111001111",
];
pub fn fixture(train: bool) -> Result<Dataset, String> {
    let variants = if train { 3 } else { 1 };
    let mut samples = Vec::new();
    for (label, glyph) in GLYPHS.iter().enumerate() {
        for variant in 0..variants {
            let high = if train { 255 - variant * 15 } else { 225 };
            let low = if train { variant * 8 } else { 20 };
            let pixels: Vec<u8> = glyph
                .bytes()
                .map(|v| if v == b'1' { high as u8 } else { low as u8 })
                .collect();
            samples.push((pixels, label as u8));
        }
    }
    if !train {
        // Two identical blank images with different labels make ambiguity observable.
        samples.push((vec![0; 15], 1));
        samples.push((vec![0; 15], 7));
    }
    let mut images = Vec::new();
    for v in [2051u32, samples.len() as u32, 5, 3] {
        images.extend(v.to_be_bytes());
    }
    let mut labels = Vec::new();
    for v in [2049u32, samples.len() as u32] {
        labels.extend(v.to_be_bytes());
    }
    for (pixels, label) in samples {
        images.extend(pixels);
        labels.push(label);
    }
    parse_idx(&images, &labels)
}
pub fn datasets(args: &[String]) -> Result<(Dataset, Dataset, usize), String> {
    match args {
        [] => Ok((fixture(true)?,fixture(false)?,160)),
        [flag,ti,tl,vi,vl,epochs] if flag=="--mnist" => {
            let epochs=epochs.parse::<usize>().map_err(|_| "epochs must be a positive integer")?;
            if !(1..=100).contains(&epochs) {return Err("explicit epochs must be in 1..=100".into());}
            let train=load_idx(ti,tl)?;let held=load_idx(vi,vl)?;
            if train.rows!=held.rows || train.cols!=held.cols {return Err("train and held-out image shapes differ".into());}
            Ok((train,held,epochs))
        },
        _ => Err("usage: NN [--solution] [--check] [--mnist TRAIN_IMAGES TRAIN_LABELS VALID_IMAGES VALID_LABELS EPOCHS]".into()),
    }
}
pub fn inspect(data: &Dataset) {
    println!(
        "{} images of {}×{}; first label={}",
        data.len(),
        data.rows,
        data.cols,
        data.labels[0]
    );
    for row in data.image(0).chunks(data.cols).take(28) {
        println!(
            "{}",
            row.iter()
                .take(28)
                .map(|&v| if v > 0.5 { '#' } else { '.' })
                .collect::<String>()
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixture_and_invalid_headers() -> Result<(), String> {
        let data = fixture(true)?;
        assert_eq!((data.len(), data.rows, data.cols), (30, 5, 3));
        assert!(parse_idx(&[], &[]).is_err());
        let mut images = Vec::new();
        for v in [2051u32, 1, 1, 1] {
            images.extend(v.to_be_bytes());
        }
        images.push(127);
        let mut labels = Vec::new();
        for v in [2049u32, 1] {
            labels.extend(v.to_be_bytes());
        }
        labels.push(9);
        assert_eq!(parse_idx(&images, &labels)?.image(0), [127. / 255.]);
        assert!(parse_idx(&images[..16], &labels).is_err());
        labels[8] = 10;
        assert!(parse_idx(&images, &labels).is_err());
        labels[8] = 0;
        images[4..8].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(parse_idx(&images, &labels).is_err());
        Ok(())
    }
}
