use std::collections::HashMap;

pub const TEXT: &str =
    "rust learns from bytes. rust learns from examples. bytes become predictions.";

pub fn byte_bigrams(data: &[u8]) -> HashMap<(u8, u8), usize> {
    let mut counts = HashMap::new();
    for pair in data.windows(2) {
        *counts.entry((pair[0], pair[1])).or_default() += 1;
    }
    counts
}

pub fn char_trigrams(text: &str) -> HashMap<String, usize> {
    let chars: Vec<char> = text.chars().collect();
    let mut counts = HashMap::new();
    for window in chars.windows(3) {
        *counts.entry(window.iter().collect()).or_default() += 1;
    }
    counts
}

pub fn cross_entropy_from_logits(logits: &[f64], target: usize) -> f64 {
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (maximum - logits[target])
        + logits
            .iter()
            .map(|logit| (logit - maximum).exp())
            .sum::<f64>()
            .ln()
}

#[derive(Clone, Debug)]
pub struct Model {
    pub width: usize,
    pub embeddings: Vec<f64>,
    pub output_weights: Vec<f64>,
    pub bias: Vec<f64>,
}

impl Model {
    pub fn new(width: usize) -> Self {
        assert!(width > 0);
        let mut seed = 33_u64;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as f64 / 16_777_216.0 - 0.5) * 0.1
        };
        Self {
            width,
            embeddings: (0..256 * width).map(|_| random()).collect(),
            output_weights: (0..width * 256).map(|_| random()).collect(),
            bias: vec![0.0; 256],
        }
    }
    pub fn logits(&self, input: u8) -> Vec<f64> {
        let mut logits = self.bias.clone();
        for h in 0..self.width {
            for (class_id, logit) in logits.iter_mut().enumerate() {
                *logit += self.embeddings[input as usize * self.width + h]
                    * self.output_weights[h * 256 + class_id];
            }
        }
        logits
    }
    pub fn probabilities(&self, input: u8) -> Vec<f64> {
        let logits = self.logits(input);
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let normalizer = logits
            .iter()
            .map(|logit| (*logit - maximum).exp())
            .sum::<f64>();
        logits
            .iter()
            .map(|logit| (*logit - maximum).exp() / normalizer)
            .collect()
    }
    pub fn loss(&self, data: &[u8]) -> f64 {
        assert!(data.len() >= 2);
        data.windows(2)
            .map(|pair| {
                let logits = self.logits(pair[0]);
                let target = pair[1] as usize;
                cross_entropy_from_logits(&logits, target)
            })
            .sum::<f64>()
            / (data.len() - 1) as f64
    }
}

pub type CountLoss = fn(&[u8], &[u8]) -> crate::LabResult<f64>;
pub type Update = fn(&mut Model, &[u8], f64) -> crate::LabResult;
pub fn run_with(args: &[String], update: Update, count: CountLoss) -> crate::LabResult {
    crate::no_args(args)?;
    let data = TEXT.as_bytes();
    let valid = b"rust predicts bytes.";
    println!(
        "units: é = {} bytes; Unicode trigrams = {}",
        "é".len(),
        char_trigrams("aé中b").len()
    );
    println!(
        "observed bigrams={} count-model validation loss={:.5}",
        byte_bigrams(data).len(),
        count(data, valid)?
    );
    let mut model = Model::new(4);
    let before = model.loss(data);
    for _ in 0..80 {
        update(&mut model, data, 0.8)?;
    }
    println!(
        "embedding model train={before:.5}->{:.5} validation={:.5}; width=4 seed=33",
        model.loss(data),
        model.loss(valid)
    );
    Ok(())
}
pub fn validate(model: &Model, data: &[u8], rate: f64) -> crate::LabResult {
    crate::ensure(
        data.len() >= 2
            && rate.is_finite()
            && rate > 0.0
            && model
                .embeddings
                .iter()
                .chain(&model.output_weights)
                .chain(&model.bias)
                .all(|v| v.is_finite()),
        "need two bytes, positive finite rate and finite model",
    )
}
pub fn count_loss(train: &[u8], data: &[u8]) -> f64 {
    let counts = byte_bigrams(train);
    data.windows(2)
        .map(|p| {
            let total: usize = counts
                .iter()
                .filter(|((x, _), _)| *x == p[0])
                .map(|(_, n)| n)
                .sum();
            -((counts.get(&(p[0], p[1])).copied().unwrap_or(0) as f64 + 1.0)
                / (total as f64 + 256.0))
                .ln()
        })
        .sum::<f64>()
        / (data.len() - 1) as f64
}
pub fn check_with(update: Update, count: CountLoss) -> crate::LabResult {
    let value = count(b"abac", b"ad")?;
    crate::ensure(
        (value - count_loss(b"abac", b"ad")).abs() < 1e-10,
        format!("goal: add-one count loss for unseen a→d must be ln(258), got {value}"),
    )?;
    let data = b"abacaba";
    let old = Model::new(3);
    let mut next = old.clone();
    update(&mut next, data, 0.01)?;
    for (group, index) in [
        (0, b'a' as usize * 3 + 1),
        (1, 256 + b'b' as usize),
        (2, b'c' as usize),
    ] {
        let mut plus = old.clone();
        let mut minus = old.clone();
        // Separate field selection avoids borrowing two model states together.
        match group {
            0 => {
                plus.embeddings[index] += 1e-5;
                minus.embeddings[index] -= 1e-5;
            }
            1 => {
                plus.output_weights[index] += 1e-5;
                minus.output_weights[index] -= 1e-5;
            }
            _ => {
                plus.bias[index] += 1e-5;
                minus.bias[index] -= 1e-5;
            }
        }
        let a = match group {
            0 => (old.embeddings[index] - next.embeddings[index]) / 0.01,
            1 => (old.output_weights[index] - next.output_weights[index]) / 0.01,
            _ => (old.bias[index] - next.bias[index]) / 0.01,
        };
        let numeric = (plus.loss(data) - minus.loss(data)) / 2e-5;
        crate::ensure((a-numeric).abs() <= 1e-6+1e-4*numeric.abs(), format!("goal: gradient group {group}: learner={a:.8}, central difference={numeric:.8}; implement embedding and output outer products before simultaneous update"))?;
    }
    for _ in 0..60 {
        update(&mut next, b"xyxyxyxy", 0.8)?;
    }
    crate::ensure(
        next.loss(b"xyxy") < old.loss(b"xyxy") - 1.0,
        "goal: unfamiliar alternating bytes should learn",
    )?;
    crate::ensure(
        next.embeddings != old.embeddings && next.output_weights != old.output_weights,
        "goal: both parameter families must train",
    )?;
    println!("33 goal passed: embedding/output/bias derivatives and unfamiliar loss reduction");
    Ok(())
}
#[cfg(test)]
mod lab_tests {
    #[test]
    fn supplied_baseline_and_solution() {
        let mut model = super::Model::new(3);
        let before = model.loss(b"abba");
        crate::ch33::update(&mut model, b"abba", 0.8).unwrap();
        assert!(model.loss(b"abba") < before);
        assert!(crate::ch33::update(&mut model, b"a", 0.8).is_err());
        super::check_with(
            crate::solutions::ch33::update,
            crate::solutions::ch33::count_loss,
        )
        .unwrap();
    }
}
