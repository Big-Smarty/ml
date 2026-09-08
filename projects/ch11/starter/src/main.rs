//! A working SGD classifier, ready to gain momentum.
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
    fn width(&self) -> usize {
        self.rows * self.cols
    }
    fn image(&self, n: usize) -> &[f64] {
        let d = self.width();
        &self.images[n * d..(n + 1) * d]
    }
}
fn u32be(bytes: &[u8], offset: usize) -> Result<usize, String> {
    Ok(u32::from_be_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or("truncated IDX header")?
            .try_into()
            .unwrap(),
    ) as usize)
}
fn parse_idx(images: &[u8], labels: &[u8]) -> Result<Dataset, String> {
    if u32be(images, 0)? != 2051 || u32be(labels, 0)? != 2049 {
        return Err("expected IDX image magic 2051 and label magic 2049".into());
    }
    let (n, nl, r, c) = (
        u32be(images, 4)?,
        u32be(labels, 4)?,
        u32be(images, 8)?,
        u32be(images, 12)?,
    );
    if n == 0 || r == 0 || c == 0 || n != nl {
        return Err("IDX counts/dimensions are invalid".into());
    }
    let z = n
        .checked_mul(r)
        .and_then(|v| v.checked_mul(c))
        .ok_or("IDX size overflow")?;
    let image_len = z.checked_add(16).ok_or("IDX image length overflow")?;
    let label_len = n.checked_add(8).ok_or("IDX label length overflow")?;
    if images.len() != image_len || labels.len() != label_len {
        return Err("IDX payload length does not match header".into());
    }
    if labels[8..].iter().any(|&v| v as usize >= CLASSES) {
        return Err("IDX label is outside 0..9".into());
    }
    Ok(Dataset {
        images: images[16..].iter().map(|&v| v as f64 / 255.).collect(),
        labels: labels[8..].to_vec(),
        rows: r,
        cols: c,
    })
}
fn load(image_path: &str, label_path: &str) -> Result<Dataset, String> {
    let images = fs::read(image_path).map_err(|e| format!("cannot read {image_path}: {e}"))?;
    let labels = fs::read(label_path).map_err(|e| format!("cannot read {label_path}: {e}"))?;
    parse_idx(&images, &labels)
}
fn fixture(train: bool) -> Dataset {
    let s = if train {
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
    let mut i = Vec::new();
    for v in [2051u32, s.len() as u32, 2, 2] {
        i.extend(v.to_be_bytes())
    }
    let mut l = Vec::new();
    for v in [2049u32, s.len() as u32] {
        l.extend(v.to_be_bytes())
    }
    for (a, y) in s {
        i.extend(a);
        l.push(y)
    }
    parse_idx(&i, &l).unwrap()
}

#[derive(Clone, Debug)]
struct Mlp {
    input: usize,
    hidden: usize,
    parameters: Vec<f64>,
}
impl Mlp {
    fn checked_count(input: usize, hidden: usize) -> Option<usize> {
        hidden
            .checked_mul(input)?
            .checked_add(hidden)?
            .checked_add(CLASSES.checked_mul(hidden)?)?
            .checked_add(CLASSES)
    }
    fn count(input: usize, hidden: usize) -> usize {
        Self::checked_count(input, hidden).expect("model dimensions are too large")
    }
    fn new(input: usize, hidden: usize) -> Self {
        let mut seed = 7u64;
        let mut parameters = vec![0.; Self::count(input, hidden)];
        let bias1_offset = hidden * input;
        let weights2_offset = bias1_offset + hidden;
        let mut draw = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((seed >> 11) as f64 / (1u64 << 53) as f64) * 2. - 1.) * scale
        };
        for weight in &mut parameters[..bias1_offset] {
            *weight = draw((6. / input as f64).sqrt())
        }
        for weight in &mut parameters[weights2_offset..weights2_offset + CLASSES * hidden] {
            *weight = draw((6. / hidden as f64).sqrt())
        }
        Self {
            input,
            hidden,
            parameters,
        }
    }
    fn parameter_offsets(&self) -> (usize, usize, usize, usize) {
        let weights1 = 0;
        let bias1 = self.hidden * self.input;
        let weights2 = bias1 + self.hidden;
        let bias2 = weights2 + CLASSES * self.hidden;
        (weights1, bias1, weights2, bias2)
    }
    fn forward(&self, image: &[f64]) -> (Vec<f64>, [f64; CLASSES]) {
        let (weights1, bias1, weights2, bias2) = self.parameter_offsets();
        let mut hidden = vec![0.; self.hidden];
        for (j, activation) in hidden.iter_mut().enumerate() {
            let mut z = self.parameters[bias1 + j];
            for (k, &input) in image.iter().enumerate() {
                z += self.parameters[weights1 + j * self.input + k] * input
            }
            *activation = z.max(0.)
        }
        let mut logits = [0.; CLASSES];
        for (c, logit) in logits.iter_mut().enumerate() {
            *logit = self.parameters[bias2 + c];
            for (j, &activation) in hidden.iter().enumerate() {
                *logit += self.parameters[weights2 + c * self.hidden + j] * activation
            }
        }
        (hidden, logits)
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
    fn loss_and_gradient(
        &self,
        data: &Dataset,
        start: usize,
        end: usize,
    ) -> Result<(f64, Vec<f64>), String> {
        if data.width() != self.input || start >= end || end > data.len() {
            return Err("invalid model/data batch".into());
        }
        let mut gradient = vec![0.; self.parameters.len()];
        let mut loss = 0.;
        let (weights1, bias1, weights2, bias2) = self.parameter_offsets();
        for n in start..end {
            let image = data.image(n);
            let (hidden, logits) = self.forward(image);
            if hidden.iter().chain(&logits).any(|value| !value.is_finite()) {
                return Err("forward computation overflowed".into());
            }
            loss += Self::cross_entropy_from_logits(logits, data.labels[n] as usize);
            let mut logit_gradient = Self::probabilities(logits);
            logit_gradient[data.labels[n] as usize] -= 1.;
            let mut hidden_gradient = vec![0.; self.hidden];
            for c in 0..CLASSES {
                gradient[bias2 + c] += logit_gradient[c];
                for j in 0..self.hidden {
                    gradient[weights2 + c * self.hidden + j] += logit_gradient[c] * hidden[j];
                    hidden_gradient[j] +=
                        logit_gradient[c] * self.parameters[weights2 + c * self.hidden + j]
                }
            }
            for j in 0..self.hidden {
                if hidden[j] > 0. {
                    gradient[bias1 + j] += hidden_gradient[j];
                    for k in 0..self.input {
                        gradient[weights1 + j * self.input + k] += hidden_gradient[j] * image[k]
                    }
                }
            }
        }
        let batch_len = (end - start) as f64;
        for value in &mut gradient {
            *value /= batch_len
        }
        if !loss.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            return Err("loss or gradient overflowed".into());
        }
        Ok((loss / batch_len, gradient))
    }
    fn metrics(&self, data: &Dataset) -> Result<(f64, f64), String> {
        if data.width() != self.input || data.len() == 0 {
            return Err("evaluation shape mismatch".into());
        }
        let (mut loss, mut good) = (0., 0usize);
        for n in 0..data.len() {
            let (_, logits) = self.forward(data.image(n));
            loss += Self::cross_entropy_from_logits(logits, data.labels[n] as usize);
            let prediction = (0..CLASSES)
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .unwrap();
            good += (prediction == data.labels[n] as usize) as usize
        }
        if !loss.is_finite() {
            return Err("evaluation loss overflowed".into());
        }
        Ok((loss / data.len() as f64, good as f64 / data.len() as f64))
    }
}

#[cfg(test)]
fn momentum_step(
    parameter: &mut f64,
    velocity: &mut f64,
    gradient: f64,
    learning_rate: f64,
    beta: f64,
) {
    let _ = (parameter, velocity, gradient, learning_rate, beta);
    // TODO: update velocity from the old velocity and gradient, then the parameter.
    todo!("implement momentum")
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    let data = match args.as_slice() {
        [_] => fixture(true),
        [_, images, labels] => load(images, labels)?,
        _ => return Err("usage: ch11-starter [IMAGES_IDX LABELS_IDX]".into()),
    };
    let mut model = Mlp::new(data.width(), 16);
    let before = model.metrics(&data)?.0;
    // A small batch and familiar SGD keep this checkpoint quick even with MNIST.
    for _ in 0..20 {
        let (_, gradient) = model.loss_and_gradient(&data, 0, data.len().min(32))?;
        for (parameter, gradient) in model.parameters.iter_mut().zip(gradient) {
            *parameter -= 0.1 * gradient;
        }
    }
    println!(
        "SGD checkpoint: training loss {before:.4} -> {:.4}",
        model.metrics(&data)?.0
    );
    println!("Complete momentum_step, allocate one velocity per parameter, and replace the SGD update above.");
    Ok(())
}
#[test]
fn first_momentum_step() {
    let (mut parameter, mut velocity) = (1.0, 0.0);
    momentum_step(&mut parameter, &mut velocity, 2.0, 0.1, 0.9);
    assert!((velocity - 2.0).abs() < 1e-12);
    assert!((parameter - 0.8).abs() < 1e-12);
    momentum_step(&mut parameter, &mut velocity, 0.0, 0.1, 0.9);
    assert!((velocity - 1.8).abs() < 1e-12);
    assert!((parameter - 0.62).abs() < 1e-12);
}
