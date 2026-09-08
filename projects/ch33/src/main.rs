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

#[derive(Clone)]
struct BigramEmbeddingLm {
    width: usize,
    embeddings: Vec<f32>,
    output: Vec<f32>,
    bias: Vec<f32>,
}

impl BigramEmbeddingLm {
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
            output: (0..width * 256).map(|_| random()).collect(),
            bias: vec![0.0; 256],
        }
    }
    fn logits(&self, byte: u8) -> Vec<f32> {
        let mut z = self.bias.clone();
        for h in 0..self.width {
            for (next, value) in z.iter_mut().enumerate() {
                *value +=
                    self.embeddings[byte as usize * self.width + h] * self.output[h * 256 + next];
            }
        }
        z
    }
    fn loss(&self, data: &[u8]) -> f32 {
        assert!(data.len() >= 2);
        data.windows(2)
            .map(|p| {
                let z = self.logits(p[0]);
                let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let s = z.iter().map(|x| (*x - m).exp()).sum::<f32>();
                s.ln() + (m - z[p[1] as usize])
            })
            .sum::<f32>()
            / (data.len() - 1) as f32
    }
    fn step(&mut self, data: &[u8], rate: f32) {
        assert!(data.len() >= 2 && rate.is_finite() && rate > 0.0);
        let mut de = vec![0.0; self.embeddings.len()];
        let mut dw = vec![0.0; self.output.len()];
        let mut db = vec![0.0; 256];
        let n = (data.len() - 1) as f32;
        for p in data.windows(2) {
            let z = self.logits(p[0]);
            let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let s = z.iter().map(|x| (*x - m).exp()).sum::<f32>();
            for j in 0..256 {
                let mut g = (z[j] - m).exp() / s;
                g -= f32::from(j == p[1] as usize);
                g /= n;
                db[j] += g;
                for h in 0..self.width {
                    let e = p[0] as usize * self.width + h;
                    de[e] += g * self.output[h * 256 + j];
                    dw[h * 256 + j] += g * self.embeddings[e];
                }
            }
        }
        for (p, g) in self.embeddings.iter_mut().zip(de) {
            *p -= rate * g;
        }
        for (p, g) in self.output.iter_mut().zip(dw) {
            *p -= rate * g;
        }
        for (p, g) in self.bias.iter_mut().zip(db) {
            *p -= rate * g;
        }
    }
    fn predict(&self, byte: u8) -> u8 {
        let z = self.logits(byte);
        z.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
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
    let mut model = BigramEmbeddingLm::new(16);
    let before = model.loss(bytes);
    for _ in 0..250 {
        model.step(bytes, 1.0);
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
        let mut model = BigramEmbeddingLm::new(3);
        model.bias.fill(1e8);
        assert!((model.loss(b"ab") - 256.0_f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn ngrams_respect_units() {
        assert_eq!(byte_bigrams("é".as_bytes()).len(), 1);
        assert!(char_trigrams("aé中b").contains_key("aé中"));
    }
    #[test]
    fn all_embedding_weights_train() {
        let data = b"abababab";
        let mut m = BigramEmbeddingLm::new(4);
        let old = m.clone();
        let before = m.loss(data);
        for _ in 0..30 {
            m.step(data, 1.0);
        }
        assert!(m.loss(data) < before);
        assert_ne!(m.embeddings, old.embeddings);
        assert_ne!(m.output, old.output);
    }
    #[test]
    fn embedding_and_output_gradients_match_finite_differences() {
        let data = b"aba";
        let mut model = BigramEmbeddingLm::new(3);
        let old = model.clone();
        model.step(data, 1e-3);
        for (group, index) in [(0, b'a' as usize * 3 + 1), (1, 256 + b'b' as usize)] {
            let analytic = if group == 0 {
                (old.embeddings[index] - model.embeddings[index]) / 1e-3
            } else {
                (old.output[index] - model.output[index]) / 1e-3
            };
            let mut plus = old.clone();
            let mut minus = old.clone();
            let eps = 1e-3;
            if group == 0 {
                plus.embeddings[index] += eps;
                minus.embeddings[index] -= eps;
            } else {
                plus.output[index] += eps;
                minus.output[index] -= eps;
            }
            let numeric = (plus.loss(data) - minus.loss(data)) / (2.0 * eps);
            assert!((analytic - numeric).abs() < 2e-3, "{analytic} != {numeric}");
        }
    }
}
