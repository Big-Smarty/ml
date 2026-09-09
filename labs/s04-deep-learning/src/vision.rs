//! A two-filter convolutional digit classifier with full backpropagation.
use std::fs;

const CLASSES: usize = 10;
const FILTERS: usize = 2;

type Convolution = fn(&[f64], usize, usize, &[f64; 9], f64) -> Vec<f64>;
type Pool = fn(&[f64], usize, usize) -> (Vec<f64>, Vec<usize>);
#[derive(Clone, Copy)]
pub(crate) struct Core {
    pub convolve: Convolution,
    pub pool: Pool,
    pub backward: fn(&[f64], usize, usize, &[f64]) -> [f64; 9],
}

#[derive(Clone)]
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
    fn width(&self) -> usize {
        self.rows * self.cols
    }
    fn image(&self, n: usize) -> &[f64] {
        let d = self.width();
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
        labels: labels[8..].to_vec(),
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
struct Cnn {
    core: Core,
    rows: usize,
    cols: usize,
    convolution_weights: [[f64; 9]; FILTERS],
    convolution_biases: [f64; FILTERS],
    dense_weights: Vec<f64>,
    class_biases: [f64; CLASSES],
}

struct ForwardCache {
    relu: Vec<f64>,
    winners: Vec<usize>,
    pooled: Vec<f64>,
    logits: [f64; CLASSES],
}

struct Gradient {
    convolution_weights: [[f64; 9]; FILTERS],
    convolution_biases: [f64; FILTERS],
    dense_weights: Vec<f64>,
    class_biases: [f64; CLASSES],
}

struct Optimizer {
    learning_rate: f64,
    step_count: u64,
}

impl Cnn {
    fn new(rows: usize, cols: usize, core: Core) -> Result<Self, String> {
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
        let mut dense_weights = vec![0.0; dense_len];
        for (i, value) in dense_weights.iter_mut().enumerate() {
            *value = ((i * 37 + 11) as f64).sin() * 0.08;
        }
        Ok(Self {
            core,
            rows,
            cols,
            convolution_weights: [
                [-0.08, 0.02, 0.08, -0.12, 0.03, 0.12, -0.08, 0.02, 0.08],
                [-0.08, -0.12, -0.08, 0.02, 0.03, 0.02, 0.08, 0.12, 0.08],
            ],
            convolution_biases: [0.02; FILTERS],
            dense_weights,
            class_biases: [0.0; CLASSES],
        })
    }

    fn forward(&self, image: &[f64]) -> ForwardCache {
        assert_eq!(image.len(), self.rows * self.cols);
        let plane = self.rows * self.cols;
        let mut relu = vec![0.0; FILTERS * plane];
        let mut pooled = Vec::new();
        let mut winners = Vec::new();
        for f in 0..FILTERS {
            let values = (self.core.convolve)(
                image,
                self.rows,
                self.cols,
                &self.convolution_weights[f],
                self.convolution_biases[f],
            );
            for (j, value) in values.iter().enumerate() {
                relu[f * plane + j] = value.max(0.0);
            }
            let (values, indices) =
                (self.core.pool)(&relu[f * plane..(f + 1) * plane], self.rows, self.cols);
            pooled.extend(values);
            winners.extend(indices.into_iter().map(|j| f * plane + j));
        }
        let mut logits = self.class_biases;
        for (y, logit) in logits.iter_mut().enumerate() {
            for (j, &value) in pooled.iter().enumerate() {
                *logit += self.dense_weights[y * pooled.len() + j] * value;
            }
        }
        ForwardCache {
            relu,
            winners,
            pooled,
            logits,
        }
    }

    fn probabilities(logits: [f64; CLASSES]) -> [f64; CLASSES] {
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut probabilities = logits.map(|logit| (logit - maximum).exp());
        let sum = probabilities.iter().sum::<f64>();
        for probability in &mut probabilities {
            *probability /= sum;
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

    fn loss_and_gradient(&self, image: &[f64], target: usize) -> (f64, Gradient) {
        let cache = self.forward(image);
        let loss = Self::cross_entropy_from_logits(cache.logits, target);
        let mut logit_gradient = Self::probabilities(cache.logits);
        logit_gradient[target] -= 1.0;
        let mut gradient = Gradient {
            convolution_weights: [[0.0; 9]; FILTERS],
            convolution_biases: [0.0; FILTERS],
            dense_weights: vec![0.0; self.dense_weights.len()],
            class_biases: logit_gradient,
        };
        let mut pooled_gradient = vec![0.0; cache.pooled.len()];
        for (class, &class_gradient) in logit_gradient.iter().enumerate() {
            for (feature, pooled_value_gradient) in pooled_gradient.iter_mut().enumerate() {
                gradient.dense_weights[class * cache.pooled.len() + feature] =
                    class_gradient * cache.pooled[feature];
                *pooled_value_gradient +=
                    self.dense_weights[class * cache.pooled.len() + feature] * class_gradient;
            }
        }
        let plane = self.rows * self.cols;
        let mut relu_gradient = vec![0.0; cache.relu.len()];
        for (j, &winner) in cache.winners.iter().enumerate() {
            relu_gradient[winner] += pooled_gradient[j];
        }
        for f in 0..FILTERS {
            let activation_gradient: Vec<f64> = (0..plane)
                .map(|j| {
                    let at = f * plane + j;
                    if cache.relu[at] > 0.0 {
                        relu_gradient[at]
                    } else {
                        0.0
                    }
                })
                .collect();
            gradient.convolution_biases[f] = activation_gradient.iter().sum();
            gradient.convolution_weights[f] =
                (self.core.backward)(image, self.rows, self.cols, &activation_gradient);
        }
        (loss, gradient)
    }

    fn train(
        &mut self,
        optimizer: &mut Optimizer,
        data: &Dataset,
        epochs: usize,
        limit: usize,
    ) -> Result<(), String> {
        if data.rows != self.rows || data.cols != self.cols || data.len() == 0 {
            return Err("training data shape does not match the model".into());
        }
        if epochs == 0 {
            return Err("epochs and learning rate must be positive".into());
        }
        let n = data.len().min(limit);
        if n == 0 {
            return Err("training limit must include at least one row".into());
        }
        for _ in 0..epochs {
            for i in 0..n {
                let (loss, gradient) =
                    self.loss_and_gradient(data.image(i), data.labels[i] as usize);
                if !loss.is_finite() {
                    return Err("non-finite training loss; reduce the learning rate".into());
                }
                optimizer.step(self, &gradient)?;
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
            let cache = self.forward(data.image(i));
            loss += Self::cross_entropy_from_logits(cache.logits, data.labels[i] as usize);
            let prediction = (0..CLASSES)
                .max_by(|&a, &b| cache.logits[a].total_cmp(&cache.logits[b]))
                .unwrap();
            correct += usize::from(prediction == data.labels[i] as usize);
        }
        if !loss.is_finite() {
            return Err("non-finite evaluation loss".into());
        }
        Ok((loss / n as f64, correct as f64 / n as f64))
    }
}

impl Optimizer {
    fn new(learning_rate: f64) -> Result<Self, String> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("epochs and learning rate must be positive".into());
        }
        Ok(Self {
            learning_rate,
            step_count: 0,
        })
    }

    fn step(&mut self, model: &mut Cnn, gradient: &Gradient) -> Result<(), String> {
        self.step_count = self
            .step_count
            .checked_add(1)
            .ok_or("optimizer step overflow")?;
        for filter in 0..FILTERS {
            for kernel_value in 0..9 {
                model.convolution_weights[filter][kernel_value] -=
                    self.learning_rate * gradient.convolution_weights[filter][kernel_value];
            }
            model.convolution_biases[filter] -=
                self.learning_rate * gradient.convolution_biases[filter];
        }
        for (weight, weight_gradient) in model.dense_weights.iter_mut().zip(&gradient.dense_weights)
        {
            *weight -= self.learning_rate * weight_gradient;
        }
        for class in 0..CLASSES {
            model.class_biases[class] -= self.learning_rate * gradient.class_biases[class];
        }
        if model
            .convolution_weights
            .iter()
            .flatten()
            .chain(&model.convolution_biases)
            .chain(&model.dense_weights)
            .chain(&model.class_biases)
            .any(|parameter| !parameter.is_finite())
        {
            return Err("training update overflow; reduce the learning rate".into());
        }
        Ok(())
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
            labels.push(digit as u8);
        }
    }
    Dataset {
        images,
        labels,
        rows: 8,
        cols: 8,
    }
}

fn experiment(
    core: Core,
    train: Dataset,
    valid: Dataset,
    epochs: usize,
    learning_rate: f64,
    limit: usize,
) -> Result<(), String> {
    if train.rows != valid.rows || train.cols != valid.cols {
        return Err("fit and validation image dimensions differ".into());
    }
    println!("epochs={epochs}, learning_rate={learning_rate}; first training label={}, first image row={:?}",train.labels[0],&train.image(0)[..train.cols]);
    let mut model = Cnn::new(train.rows, train.cols, core)?;
    let mut optimizer = Optimizer::new(learning_rate)?;
    let before = model.metrics(&valid, limit)?;
    model.train(&mut optimizer, &train, epochs, limit)?;
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

pub(crate) fn run(core: Core, args: &[String]) -> Result<(), String> {
    if let [flag, stage] = args {
        if flag == "--checkpoint" {
            return check_stage(core, stage);
        }
    }
    println!("digit pipeline: selected spatial core -> ReLU -> 2x2 reduction -> 10 logits");
    match args {
        [] => experiment(core, fixture(true), fixture(false), 80, 0.025, usize::MAX),
        [flag, fi, fl, vi, vl] if flag == "--mnist" => {
            experiment(core, load_idx(fi, fl)?, load_idx(vi, vl)?, 1, 0.01, 2_000)
        }
        _ => Err("usage: 20 [--mnist FIT_IMAGES FIT_LABELS VALID_IMAGES VALID_LABELS]".into()),
    }
}

pub(crate) fn check(core: Core) -> Result<(), String> {
    check_stage(core, "all")
}
fn check_stage(core: Core, stage: &str) -> Result<(), String> {
    if !["convolution", "pooling", "gradients", "training", "all"].contains(&stage) {
        return Err(format!("unknown chapter20 checkpoint {stage}"));
    }

    let image: Vec<f64> = (1..=20).map(f64::from).collect();
    let kernel = [1.0, 2.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0];
    let values = (core.convolve)(&image, 4, 5, &kernel, 0.25);
    if values.len() != 20 || (values[6] - 1.75).abs() > 1e-12 || (values[0] + 0.25).abs() > 1e-12 {
        return Err(format!("GOAL_NOT_MET: spatial convolution: expected positions 0=-0.25, 6=1.75, got {values:?}; share all nine weights and handle padding"));
    }
    if stage == "convolution" {
        println!("PASS convolution checkpoint");
        return Ok(());
    }
    let (pooled, winners) = (core.pool)(&[0.2, 1.1, 0.7, 0.4], 2, 2);
    if pooled != [1.1] || winners != [1] {
        return Err(format!(
            "GOAL_NOT_MET: max pool: expected value 1.1 at flat index1, got {pooled:?}, {winners:?}"
        ));
    }
    if stage == "pooling" {
        println!("PASS pooling checkpoint");
        return Ok(());
    }
    // Smooth linear objective dot(convolve(image), incoming): independent central differences.
    let incoming: Vec<f64> = (0..20).map(|i| (i as f64 * 0.17).cos()).collect();
    let gradient = (core.backward)(&image, 4, 5, &incoming);
    for k in 0..9 {
        let mut plus = kernel;
        let mut minus = kernel;
        plus[k] += 1e-5;
        minus[k] -= 1e-5;
        let objective = |w: &[f64; 9]| {
            (core.convolve)(&image, 4, 5, w, 0.25)
                .iter()
                .zip(&incoming)
                .map(|(a, b)| a * b)
                .sum::<f64>()
        };
        crate::close(
            "shared convolution gradient",
            gradient[k],
            (objective(&plus) - objective(&minus)) / 2e-5,
        )?;
    }
    if stage == "gradients" {
        println!("PASS gradients checkpoint");
        return Ok(());
    }
    let mut model = Cnn::new(8, 8, core)?;
    let valid = fixture(false);
    let before = model.metrics(&valid, usize::MAX)?.0;
    let mut optimizer = Optimizer::new(0.025)?;
    model.train(&mut optimizer, &fixture(true), 80, usize::MAX)?;
    let after = model.metrics(&valid, usize::MAX)?;
    if after.0 >= before * 0.7 || after.1 < 0.7 {
        return Err(format!(
            "GOAL_NOT_MET: CNN validation: {before:.4} -> {after:?}"
        ));
    }
    println!("PASS spatial values, max route, nine kernel gradients, 4x5 transfer, CNN loss {before:.4} -> {:.4}, accuracy {:.1}%", after.0,after.1*100.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cross_entropy_preserves_small_loss_at_large_common_offset() {
        let mut model = Cnn::new(8, 8, crate::solutions::ch20::CORE).unwrap();
        model.dense_weights.fill(0.0);
        model.class_biases.fill(1e16);
        let data = fixture(false);
        let expected = (CLASSES as f64).ln();
        assert!(
            (model
                .loss_and_gradient(data.image(0), data.labels[0] as usize)
                .0
                - expected)
                .abs()
                < 1e-12
        );
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
        let mut model = Cnn::new(8, 8, crate::solutions::ch20::CORE).unwrap();
        let mut x: Vec<f64> = (0..64).map(|i| (i as f64 * 0.071).sin() + 0.3).collect();
        for (i, value) in x.iter_mut().enumerate() {
            *value += i as f64 * 1e-4;
        }
        let analytic = model.loss_and_gradient(&x, 3).1.convolution_weights[0][0];
        let h = 1e-5;
        model.convolution_weights[0][0] += h;
        let plus = model.loss_and_gradient(&x, 3).0;
        model.convolution_weights[0][0] -= 2.0 * h;
        let minus = model.loss_and_gradient(&x, 3).0;
        let numeric = (plus - minus) / (2.0 * h);
        assert!((analytic - numeric).abs() < 1e-6 + 1e-4 * numeric.abs());
    }

    #[test]
    fn training_reduces_held_out_loss() {
        let train = fixture(true);
        let valid = fixture(false);
        let mut model = Cnn::new(8, 8, crate::solutions::ch20::CORE).unwrap();
        let mut optimizer = Optimizer::new(0.025).unwrap();
        let before = model.metrics(&valid, usize::MAX).unwrap().0;
        model.train(&mut optimizer, &train, 80, usize::MAX).unwrap();
        let after = model.metrics(&valid, usize::MAX).unwrap();
        assert_eq!(optimizer.step_count, 80 * train.len() as u64);
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
        assert!(Cnn::new(usize::MAX, usize::MAX, crate::solutions::ch20::CORE).is_err());
        assert!(Cnn::new(8, 8, crate::solutions::ch20::CORE)
            .unwrap()
            .metrics(&valid, 0)
            .is_err());
        assert!(Optimizer::new(0.0).is_err());
    }
}
