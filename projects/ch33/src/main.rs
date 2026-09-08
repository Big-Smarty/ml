use std::collections::HashMap;

const TEXT: &str = "rust learns from bytes. rust learns from examples. bytes become predictions.";

fn byte_bigrams(data: &[u8]) -> HashMap<(u8, u8), usize> {
    let mut counts = HashMap::new();
    for pair in data.windows(2) {
        *counts.entry((pair[0], pair[1])).or_default() += 1;
    }
    counts
}

fn char_trigrams(text: &str) -> HashMap<String, usize> {
    let chars: Vec<char> = text.chars().collect();
    let mut counts = HashMap::new();
    for window in chars.windows(3) {
        *counts.entry(window.iter().collect()).or_default() += 1;
    }
    counts
}

fn cross_entropy_from_logits(logits: &[f32], target: usize) -> f32 {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (maximum - logits[target])
        + logits
            .iter()
            .map(|logit| (logit - maximum).exp())
            .sum::<f32>()
            .ln()
}

#[derive(Clone)]
struct Model {
    width: usize,
    embeddings: Vec<f32>,
    output_weights: Vec<f32>,
    bias: Vec<f32>,
}

impl Model {
    fn new(width: usize) -> Self {
        assert!(width > 0);
        let mut seed = 33_u64;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as f32 / 16_777_216.0 - 0.5) * 0.1
        };
        Self {
            width,
            embeddings: (0..256 * width).map(|_| random()).collect(),
            output_weights: (0..width * 256).map(|_| random()).collect(),
            bias: vec![0.0; 256],
        }
    }
    fn logits(&self, input: u8) -> Vec<f32> {
        let mut logits = self.bias.clone();
        for h in 0..self.width {
            for (class_id, logit) in logits.iter_mut().enumerate() {
                *logit += self.embeddings[input as usize * self.width + h]
                    * self.output_weights[h * 256 + class_id];
            }
        }
        logits
    }
    fn probabilities(&self, input: u8) -> Vec<f32> {
        let logits = self.logits(input);
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let normalizer = logits
            .iter()
            .map(|logit| (*logit - maximum).exp())
            .sum::<f32>();
        logits
            .iter()
            .map(|logit| (*logit - maximum).exp() / normalizer)
            .collect()
    }
    fn loss(&self, data: &[u8]) -> f32 {
        assert!(data.len() >= 2);
        data.windows(2)
            .map(|pair| {
                let logits = self.logits(pair[0]);
                let target = pair[1] as usize;
                cross_entropy_from_logits(&logits, target)
            })
            .sum::<f32>()
            / (data.len() - 1) as f32
    }
    fn step(&mut self, data: &[u8], learning_rate: f32) {
        assert!(data.len() >= 2 && learning_rate.is_finite() && learning_rate > 0.0);
        let mut embedding_gradient = vec![0.0; self.embeddings.len()];
        let mut output_gradient = vec![0.0; self.output_weights.len()];
        let mut bias_gradient = vec![0.0; 256];
        let example_count = (data.len() - 1) as f32;
        for pair in data.windows(2) {
            let input = pair[0];
            let target = pair[1] as usize;
            let probabilities = self.probabilities(input);
            for class_id in 0..256 {
                let logit_gradient =
                    (probabilities[class_id] - f32::from(class_id == target)) / example_count;
                bias_gradient[class_id] += logit_gradient;
                for h in 0..self.width {
                    let embedding_index = input as usize * self.width + h;
                    let output_index = h * 256 + class_id;
                    embedding_gradient[embedding_index] +=
                        logit_gradient * self.output_weights[output_index];
                    output_gradient[output_index] +=
                        logit_gradient * self.embeddings[embedding_index];
                }
            }
        }
        for (parameter, gradient) in self.embeddings.iter_mut().zip(embedding_gradient) {
            *parameter -= learning_rate * gradient;
        }
        for (parameter, gradient) in self.output_weights.iter_mut().zip(output_gradient) {
            *parameter -= learning_rate * gradient;
        }
        for (parameter, gradient) in self.bias.iter_mut().zip(bias_gradient) {
            *parameter -= learning_rate * gradient;
        }
    }
    fn predict(&self, input: u8) -> u8 {
        self.logits(input)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("invariant: logits contains one score per byte class")
            .0 as u8
    }
}

fn main() {
    let bytes = TEXT.as_bytes();
    let counts = byte_bigrams(bytes);
    println!(
        "byte bigram 'r' -> 'u' occurs {} times",
        counts.get(&(b'r', b'u')).unwrap_or(&0)
    );
    println!(
        "{} distinct Unicode character trigrams",
        char_trigrams("café 咖啡 café").len()
    );
    let mut model = Model::new(16);
    let before = model.loss(bytes);
    let learning_rate = 1.0;
    for _ in 0..250 {
        model.step(bytes, learning_rate);
    }
    println!(
        "next-byte cross-entropy: {before:.3} -> {:.3}",
        model.loss(bytes)
    );
    println!(
        "embedding model predicts after 'r': {:?}",
        model.predict(b'r') as char
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn common_logit_offset_preserves_uniform_loss() {
        let mut model = Model::new(3);
        model.bias.fill(1e8);
        assert!((model.loss(b"ab") - 256.0_f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn ngrams_respect_units() {
        assert_eq!(byte_bigrams("é".as_bytes()).len(), 1);
        assert!(char_trigrams("aé中b").contains_key("aé中"));
    }
    #[test]
    fn embedding_and_output_parameters_train() {
        let data = b"abababab";
        let mut model = Model::new(4);
        let old = model.clone();
        let before = model.loss(data);
        for _ in 0..30 {
            model.step(data, 1.0);
        }
        assert!(model.loss(data) < before);
        assert_ne!(model.embeddings, old.embeddings);
        assert_ne!(model.output_weights, old.output_weights);
    }
    #[test]
    fn embedding_and_output_gradients_match_finite_differences() {
        let data = b"aba";
        let mut model = Model::new(3);
        let old = model.clone();
        model.step(data, 1e-3);
        for (group, index) in [(0, b'a' as usize * 3 + 1), (1, 256 + b'b' as usize)] {
            let analytic = if group == 0 {
                (old.embeddings[index] - model.embeddings[index]) / 1e-3
            } else {
                (old.output_weights[index] - model.output_weights[index]) / 1e-3
            };
            let mut plus = old.clone();
            let mut minus = old.clone();
            let eps = 1e-3;
            if group == 0 {
                plus.embeddings[index] += eps;
                minus.embeddings[index] -= eps;
            } else {
                plus.output_weights[index] += eps;
                minus.output_weights[index] -= eps;
            }
            let numeric = (plus.loss(data) - minus.loss(data)) / (2.0 * eps);
            assert!((analytic - numeric).abs() < 2e-3, "{analytic} != {numeric}");
        }
    }
}
