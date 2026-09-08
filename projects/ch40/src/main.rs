//! Incremental decoding for the Chapter 36 decoder, with real per-layer KV caches.
use ch36::{Config, Decoder, ParameterSpan};
use std::{collections::HashMap, error::Error, io::Write, ops::Range};

#[derive(Default)]
struct LayerKv {
    keys: Vec<f32>,
    values: Vec<f32>,
}

struct CachedDecoder<'a> {
    model: &'a Decoder,
    spans: HashMap<String, Range<usize>>,
    layers: Vec<LayerKv>,
    next_position: usize,
}

impl<'a> CachedDecoder<'a> {
    fn new(model: &'a Decoder) -> Self {
        let spans = model
            .parameter_spans()
            .into_iter()
            .map(|s: ParameterSpan| (s.name, s.start..s.end))
            .collect();
        let layers = (0..model.config().layers)
            .map(|_| LayerKv::default())
            .collect();
        Self {
            model,
            spans,
            layers,
            next_position: 0,
        }
    }
    fn parameter_slice(&self, name: &str) -> &[f32] {
        &self.model.parameters()[self.spans[name].clone()]
    }
    /// Cache one token at the next sequence position and return its [vocabulary] logits.
    /// This decoding step advances no optimizer state.
    fn step(&mut self, token_id: usize) -> Result<Vec<f32>, Box<dyn Error>> {
        let c = self.model.config();
        if token_id >= c.vocab_size {
            return Err("token exceeds vocabulary".into());
        }
        if self.next_position >= c.context {
            return Err("cache reached model context; reset and replay a cropped window".into());
        }
        let mut x = vec![0.0; c.width];
        let tok = self.parameter_slice("token_embedding");
        let pos = self.parameter_slice("position_embedding");
        for j in 0..c.width {
            x[j] = tok[token_id * c.width + j] + pos[self.next_position * c.width + j];
        }
        for layer in 0..c.layers {
            let n1 = layer_norm(
                &x,
                self.parameter_slice(&format!("layer.{layer}.ln1_gain")),
                self.parameter_slice(&format!("layer.{layer}.ln1_bias")),
            );
            let mut qkv = matvec(
                &n1,
                self.parameter_slice(&format!("layer.{layer}.qkv_weight")),
                3 * c.width,
            );
            add(
                &mut qkv,
                self.parameter_slice(&format!("layer.{layer}.qkv_bias")),
            );
            let (q, rest) = qkv.split_at(c.width);
            let (k, v) = rest.split_at(c.width);
            self.layers[layer].keys.extend_from_slice(k);
            self.layers[layer].values.extend_from_slice(v);
            let mut context = vec![0.0; c.width];
            let head_dim = c.width / c.heads;
            for h in 0..c.heads {
                let mut scores = Vec::with_capacity(self.next_position + 1);
                for cached_position in 0..=self.next_position {
                    let dot: f32 = (0..head_dim)
                        .map(|j| {
                            q[h * head_dim + j]
                                * self.layers[layer].keys
                                    [cached_position * c.width + h * head_dim + j]
                        })
                        .sum();
                    scores.push(dot / (head_dim as f32).sqrt());
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denom: f32 = scores.iter().map(|s| (*s - max).exp()).sum();
                for (cached_position, score) in scores.into_iter().enumerate() {
                    let probability = (score - max).exp() / denom;
                    for j in 0..head_dim {
                        context[h * head_dim + j] += probability
                            * self.layers[layer].values
                                [cached_position * c.width + h * head_dim + j];
                    }
                }
            }
            let mut attn = matvec(
                &context,
                self.parameter_slice(&format!("layer.{layer}.attention_output_weight")),
                c.width,
            );
            add(
                &mut attn,
                self.parameter_slice(&format!("layer.{layer}.attention_output_bias")),
            );
            for j in 0..c.width {
                x[j] += attn[j];
            }
            let n2 = layer_norm(
                &x,
                self.parameter_slice(&format!("layer.{layer}.ln2_gain")),
                self.parameter_slice(&format!("layer.{layer}.ln2_bias")),
            );
            let mut ff = matvec(
                &n2,
                self.parameter_slice(&format!("layer.{layer}.ff1_weight")),
                c.ff_width,
            );
            add(
                &mut ff,
                self.parameter_slice(&format!("layer.{layer}.ff1_bias")),
            );
            ff.iter_mut().for_each(|z| *z = gelu(*z));
            let mut projected = matvec(
                &ff,
                self.parameter_slice(&format!("layer.{layer}.ff2_weight")),
                c.width,
            );
            add(
                &mut projected,
                self.parameter_slice(&format!("layer.{layer}.ff2_bias")),
            );
            for j in 0..c.width {
                x[j] += projected[j];
            }
        }
        let n = layer_norm(
            &x,
            self.parameter_slice("final_norm_gain"),
            self.parameter_slice("final_norm_bias"),
        );
        let mut logits = matvec(&n, self.parameter_slice("output_weight"), c.vocab_size);
        add(&mut logits, self.parameter_slice("output_bias"));
        self.next_position += 1;
        Ok(logits)
    }
    fn cached_values(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.keys.len() + l.values.len())
            .sum()
    }
}

fn matvec(input: &[f32], weights: &[f32], out_features: usize) -> Vec<f32> {
    (0..out_features)
        .map(|j| {
            (0..input.len())
                .map(|i| input[i] * weights[i * out_features + j])
                .sum()
        })
        .collect()
}
fn add(x: &mut [f32], b: &[f32]) {
    for (a, b) in x.iter_mut().zip(b) {
        *a += b;
    }
}
fn layer_norm(x: &[f32], gain: &[f32], bias: &[f32]) -> Vec<f32> {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (var + 1e-5).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - mean) * inv * gain[i] + bias[i])
        .collect()
}
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

#[derive(Clone)]
struct SamplingRng(u64);
impl SamplingRng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 40) as u32 as f32) / (1u32 << 24) as f32
    }
}
fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    rng: &mut SamplingRng,
) -> Result<usize, &'static str> {
    if logits.is_empty()
        || logits.iter().any(|x| !x.is_finite())
        || !temperature.is_finite()
        || temperature <= 0.0
        || top_k == 0
    {
        return Err("sampling needs logits, positive temperature, and top_k");
    };
    let mut token_ids: Vec<usize> = (0..logits.len()).collect();
    token_ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    token_ids.truncate(top_k.min(token_ids.len()));
    let max = token_ids
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let total: f32 = token_ids
        .iter()
        .map(|&i| ((logits[i] - max) / temperature).exp())
        .sum();
    let mut draw = rng.next() * total;
    for &i in &token_ids {
        draw -= ((logits[i] - max) / temperature).exp();
        if draw <= 0.0 {
            return Ok(i);
        }
    }
    Ok(*token_ids.last().unwrap())
}

struct Request<'a> {
    id: &'static str,
    tokens: Vec<usize>,
    remaining_tokens: usize,
    cache: CachedDecoder<'a>,
    logits: Vec<f32>,
    rng: SamplingRng,
}
fn run_request_batch(
    model: &Decoder,
    prompts: &[(&'static str, Vec<usize>, usize)],
    mut emit: impl FnMut(&str, usize),
) -> Result<Vec<(String, usize)>, Box<dyn Error>> {
    let mut requests = Vec::new();
    for (id, prompt, tokens_to_generate) in prompts {
        let mut cache = CachedDecoder::new(model);
        let mut logits = Vec::new();
        for &token_id in prompt {
            logits = cache.step(token_id)?;
        }
        if logits.is_empty() {
            return Err("prompt must not be empty".into());
        }
        requests.push(Request {
            id,
            tokens: prompt.clone(),
            remaining_tokens: *tokens_to_generate,
            cache,
            logits,
            rng: SamplingRng(id.len() as u64 + 9),
        });
    }
    let mut events = Vec::new();
    while requests.iter().any(|request| request.remaining_tokens > 0) {
        for request in requests
            .iter_mut()
            .filter(|request| request.remaining_tokens > 0)
        {
            let token_id = sample_token(&request.logits, 0.8, 4, &mut request.rng)?;
            request.tokens.push(token_id);
            request.remaining_tokens -= 1;
            events.push((request.id.to_string(), token_id));
            emit(request.id, token_id);
            if request.remaining_tokens > 0 {
                request.logits = request.cache.step(token_id)?;
            }
        }
    }
    Ok(events)
}

fn tiny_model() -> Result<Decoder, ch36::ModelError> {
    Decoder::new(
        Config {
            vocab_size: 12,
            context: 12,
            width: 8,
            heads: 2,
            layers: 2,
            ff_width: 16,
        },
        40,
    )
}
fn max_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() -> Result<(), Box<dyn Error>> {
    let model = tiny_model()?;
    let tokens = [1, 4, 2, 7, 3];
    let mut cached = CachedDecoder::new(&model);
    let mut worst = 0.0_f32;
    for end in 1..=tokens.len() {
        let row = cached.step(tokens[end - 1])?;
        let full = model.forward(&tokens[..end])?;
        worst = worst.max(max_error(
            &row,
            &full[full.len() - model.config().vocab_size..],
        ));
    }
    println!("cached/full worst prefix-logit error: {worst:.8}");
    println!(
        "stored K+V scalars after {} tokens: {}",
        tokens.len(),
        cached.cached_values()
    );
    run_request_batch(
        &model,
        &[("short", vec![1, 2], 2), ("long", vec![3, 4, 5], 3)],
        |id, token_id| {
            println!("stream {id}: token {token_id}");
            std::io::stdout().flush().expect("flush streamed token");
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_matches_every_full_prefix() {
        let model = tiny_model().unwrap();
        let tokens = [2, 9, 1, 6, 4, 3];
        let mut cached = CachedDecoder::new(&model);
        for end in 1..=tokens.len() {
            let cached_logits = cached.step(tokens[end - 1]).unwrap();
            let full_logits = model.forward(&tokens[..end]).unwrap();
            let expected = &full_logits[full_logits.len() - model.config().vocab_size..];
            assert_eq!(cached_logits.len(), model.config().vocab_size);
            assert!(max_error(&cached_logits, expected) < 2e-5, "prefix {end}");
        }
        assert_eq!(
            cached.cached_values(),
            2 * model.config().layers * tokens.len() * model.config().width
        );
    }
    #[test]
    fn top_k_one_is_greedy() {
        let mut rng = SamplingRng(1);
        for _ in 0..10 {
            assert_eq!(sample_token(&[1.0, 4.0, 2.0], 0.7, 1, &mut rng).unwrap(), 1);
        }
        assert!(sample_token(&[f32::NAN], 1.0, 1, &mut rng).is_err());
    }
    #[test]
    fn context_limit_is_explicit() {
        let model = tiny_model().unwrap();
        let mut cached = CachedDecoder::new(&model);
        for _ in 0..model.config().context {
            cached.step(1).unwrap();
        }
        assert!(cached.step(1).is_err());
    }
    #[test]
    fn unequal_requests_finish_and_emit_in_rounds() {
        let model = tiny_model().unwrap();
        let mut streamed = Vec::new();
        let events = run_request_batch(
            &model,
            &[("a", vec![1], 1), ("bb", vec![2], 3)],
            |id, token_id| streamed.push((id.to_string(), token_id)),
        )
        .unwrap();
        assert_eq!(events, streamed);
        assert_eq!(
            events.iter().map(|x| x.0.as_str()).collect::<Vec<_>>(),
            ["a", "bb", "bb", "bb"]
        );
    }
}
