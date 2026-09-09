// A tiny contextual decoder whose dense feed-forward block is replaced by sparse experts.
// The row-major layout and explicit backward pass continue the course-owned Chapter 36 model.
use std::{
    error::Error,
    fmt, fs,
    io::{Read, Write},
    net::TcpListener,
    ops::Range,
    path::Path,
};

const MAGIC: &[u8; 8] = b"CH56MOE2";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub context: usize,
    pub width: usize,
    pub ff_width: usize,
    pub experts: usize,
    pub capacity_factor: f64,
    pub balance_weight: f64,
}

impl Config {
    pub fn tiny(experts: usize) -> Self {
        Self {
            vocab_size: 128,
            context: 16,
            width: 8,
            ff_width: 12,
            experts,
            capacity_factor: 1.25,
            balance_weight: if experts == 1 { 0.0 } else { 0.01 },
        }
    }

    fn validate(self) -> Result<(), ModelError> {
        if self.vocab_size < 2
            || self.context == 0
            || self.width == 0
            || self.ff_width == 0
            || self.experts == 0
            || !self.capacity_factor.is_finite()
            || self.capacity_factor <= 0.0
            || !self.balance_weight.is_finite()
            || self.balance_weight < 0.0
        {
            return Err(ModelError("invalid model dimensions or routing settings"));
        }
        // Bound both parameter storage and the quadratic attention cache before allocating.
        let d = self.width;
        let e = self.experts;
        let f = self.ff_width;
        let sizes = [
            self.vocab_size.checked_mul(d),
            self.context.checked_mul(d),
            d.checked_mul(d).and_then(|n| n.checked_mul(4)),
            d.checked_mul(e),
            d.checked_mul(f)
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| n.checked_add(f))
                .and_then(|n| n.checked_add(d))
                .and_then(|n| n.checked_mul(e)),
            self.vocab_size.checked_mul(d),
            self.context.checked_mul(self.context),
            self.context.checked_mul(self.vocab_size),
            self.context.checked_mul(e),
            self.context.checked_mul(f),
            d.checked_mul(12),
            Some(e),
            Some(self.vocab_size),
        ];
        let mut budget = 0usize;
        for size in sizes {
            budget = budget
                .checked_add(size.ok_or(ModelError("dimension overflow"))?)
                .ok_or(ModelError("dimension overflow"))?;
        }
        if budget > 8_000_000 {
            return Err(ModelError(
                "reference model exceeds eight million storage elements",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ModelError(pub &'static str);

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl Error for ModelError {}

#[derive(Clone, Debug)]
struct ExpertLayout {
    w1: Range<usize>,
    b1: Range<usize>,
    w2: Range<usize>,
    b2: Range<usize>,
}

#[derive(Clone, Debug)]
struct Layout {
    norm1: Range<usize>,
    norm2: Range<usize>,
    norm_final: Range<usize>,
    token: Range<usize>,
    position: Range<usize>,
    qkv_w: Range<usize>,
    qkv_b: Range<usize>,
    attention_w: Range<usize>,
    attention_b: Range<usize>,
    router_w: Range<usize>,
    router_b: Range<usize>,
    experts: Vec<ExpertLayout>,
    output_w: Range<usize>,
    output_b: Range<usize>,
    total: usize,
}

fn take(cursor: &mut usize, len: usize) -> Range<usize> {
    let start = *cursor;
    *cursor += len;
    start..*cursor
}

fn capacity(tokens: usize, experts: usize, factor: f64) -> usize {
    assert!(experts > 0 && factor.is_finite() && factor > 0.0);
    (((tokens as f64 / experts as f64) * factor).ceil() as usize).max(1)
}

impl Layout {
    fn new(c: Config) -> Self {
        let (v, d, f, e) = (c.vocab_size, c.width, c.ff_width, c.experts);
        let mut p = 0;
        let token = take(&mut p, v * d);
        let position = take(&mut p, c.context * d);
        let norm1 = take(&mut p, 2 * d);
        let norm2 = take(&mut p, 2 * d);
        let norm_final = take(&mut p, 2 * d);
        let qkv_w = take(&mut p, d * 3 * d);
        let qkv_b = take(&mut p, 3 * d);
        let attention_w = take(&mut p, d * d);
        let attention_b = take(&mut p, d);
        let router_w = take(&mut p, if e == 1 { 0 } else { d * e });
        let router_b = take(&mut p, if e == 1 { 0 } else { e });
        let experts = (0..e)
            .map(|_| ExpertLayout {
                w1: take(&mut p, d * f),
                b1: take(&mut p, f),
                w2: take(&mut p, f * d),
                b2: take(&mut p, d),
            })
            .collect();
        let output_w = take(&mut p, d * v);
        let output_b = take(&mut p, v);
        Self {
            norm1,
            norm2,
            norm_final,
            token,
            position,
            qkv_w,
            qkv_b,
            attention_w,
            attention_b,
            router_w,
            router_b,
            experts,
            output_w,
            output_b,
            total: p,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    config: Config,
    layout: Layout,
    parameters: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Gradients {
    pub loss: f64,
    pub task_loss: f64,
    pub auxiliary_loss: f64,
    pub values: Vec<f64>,
    pub routes: Vec<usize>,
    pub accepted: Vec<usize>,
    pub dropped: usize,
}

#[derive(Clone, Debug)]
pub struct RoutingStats {
    pub attempted: Vec<usize>,
    pub accepted: Vec<usize>,
    pub dropped: usize,
    pub mean_selected_gate: f64,
    pub capacity: usize,
}

#[derive(Clone)]
struct Cache {
    norm1: NormCache,
    norm2: NormCache,
    norm_final: NormCache,
    qkv: Vec<f64>,
    attention_probs: Vec<f64>,
    context: Vec<f64>,
    router_probs: Vec<f64>,
    routes: Vec<usize>,
    accepted_mask: Vec<bool>,
    attempted: Vec<usize>,
    accepted: Vec<usize>,
    hidden: Vec<f64>,
    hidden_pre: Vec<f64>,
    expert_output: Vec<f64>,
    logits: Vec<f64>,
    auxiliary_loss: f64,
    capacity: usize,
}

impl Model {
    pub fn new(config: Config, seed: u64) -> Result<Self, ModelError> {
        config.validate()?;
        let layout = Layout::new(config);
        let mut rng = Rng::new(seed);
        let mut parameters = vec![0.0; layout.total];
        for parameter in &mut parameters {
            *parameter = (rng.next_f64() * 2.0 - 1.0) * 0.05;
        }
        for range in [&layout.norm1, &layout.norm2, &layout.norm_final] {
            parameters[range.start..range.start + config.width].fill(1.0);
            parameters[range.start + config.width..range.end].fill(0.0);
        }
        parameters[layout.qkv_b.clone()].fill(0.0);
        parameters[layout.attention_b.clone()].fill(0.0);
        parameters[layout.router_b.clone()].fill(0.0);
        parameters[layout.output_b.clone()].fill(0.0);
        for expert in &layout.experts {
            parameters[expert.b1.clone()].fill(0.0);
            parameters[expert.b2.clone()].fill(0.0);
        }
        Ok(Self {
            config,
            layout,
            parameters,
        })
    }

    pub fn config(&self) -> Config {
        self.config
    }

    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    pub fn parameters_mut(&mut self) -> &mut [f64] {
        &mut self.parameters
    }

    pub fn total_parameters(&self) -> usize {
        self.parameters.len()
    }

    pub fn active_parameters_per_token(&self) -> usize {
        let c = self.config;
        let one_expert = c.width * c.ff_width + c.ff_width + c.ff_width * c.width + c.width;
        self.parameters.len() - (c.experts - 1) * one_expert
    }

    pub fn parameter_spans(&self) -> Vec<(String, Range<usize>)> {
        let mut spans = vec![
            ("norm1".into(), self.layout.norm1.clone()),
            ("norm2".into(), self.layout.norm2.clone()),
            ("norm_final".into(), self.layout.norm_final.clone()),
            ("token_embedding".into(), self.layout.token.clone()),
            ("position_embedding".into(), self.layout.position.clone()),
            ("qkv_weight".into(), self.layout.qkv_w.clone()),
            ("qkv_bias".into(), self.layout.qkv_b.clone()),
            (
                "attention_output_weight".into(),
                self.layout.attention_w.clone(),
            ),
            (
                "attention_output_bias".into(),
                self.layout.attention_b.clone(),
            ),
            ("router_weight".into(), self.layout.router_w.clone()),
            ("router_bias".into(), self.layout.router_b.clone()),
        ];
        for (i, expert) in self.layout.experts.iter().enumerate() {
            spans.extend([
                (format!("expert.{i}.ff1_weight"), expert.w1.clone()),
                (format!("expert.{i}.ff1_bias"), expert.b1.clone()),
                (format!("expert.{i}.ff2_weight"), expert.w2.clone()),
                (format!("expert.{i}.ff2_bias"), expert.b2.clone()),
            ]);
        }
        spans.extend([
            ("output_weight".into(), self.layout.output_w.clone()),
            ("output_bias".into(), self.layout.output_b.clone()),
        ]);
        spans
    }

    fn validate_tokens(&self, tokens: &[usize]) -> Result<(), ModelError> {
        if tokens.is_empty() || tokens.len() > self.config.context {
            return Err(ModelError("token length must be in 1..=context"));
        }
        if tokens.iter().any(|&token| token >= self.config.vocab_size) {
            return Err(ModelError("token id exceeds vocabulary"));
        }
        Ok(())
    }

    fn forward_cached(&self, tokens: &[usize], capped: bool) -> Result<Cache, ModelError> {
        self.validate_tokens(tokens)?;
        let c = self.config;
        let (t, d, v) = (tokens.len(), c.width, c.vocab_size);
        let mut x = vec![0.0; t * d];
        for i in 0..t {
            for j in 0..d {
                x[i * d + j] = self.parameters[self.layout.token.start + tokens[i] * d + j]
                    + self.parameters[self.layout.position.start + i * d + j];
            }
        }
        let norm1 = layer_norm(&x, &self.parameters[self.layout.norm1.clone()], d);
        let mut qkv = vec![0.0; t * 3 * d];
        matmul_bias(
            &norm1.output,
            &self.parameters[self.layout.qkv_w.clone()],
            &self.parameters[self.layout.qkv_b.clone()],
            &mut qkv,
            t,
            d,
            3 * d,
        );
        let mut attention_probs = vec![0.0; t * t];
        let mut context = vec![0.0; t * d];
        let scale = (d as f64).sqrt();
        for i in 0..t {
            let mut scores = vec![0.0; i + 1];
            for j in 0..=i {
                scores[j] = (0..d)
                    .map(|k| qkv[i * 3 * d + k] * qkv[j * 3 * d + d + k])
                    .sum::<f64>()
                    / scale;
            }
            softmax_in_place(&mut scores);
            for j in 0..=i {
                attention_probs[i * t + j] = scores[j];
                for k in 0..d {
                    context[i * d + k] += scores[j] * qkv[j * 3 * d + 2 * d + k];
                }
            }
        }
        let mut attention = vec![0.0; t * d];
        matmul_bias(
            &context,
            &self.parameters[self.layout.attention_w.clone()],
            &self.parameters[self.layout.attention_b.clone()],
            &mut attention,
            t,
            d,
            d,
        );
        let residual: Vec<_> = x.iter().zip(attention).map(|(a, b)| a + b).collect();
        let norm2 = layer_norm(&residual, &self.parameters[self.layout.norm2.clone()], d);
        let MoeCache {
            router_probs,
            routes,
            accepted_mask,
            attempted,
            accepted,
            hidden,
            hidden_pre,
            expert_output,
            block_output,
            auxiliary_loss,
            capacity,
        } = self.moe_forward(&norm2, &residual, capped);
        let norm_final = layer_norm(
            &block_output,
            &self.parameters[self.layout.norm_final.clone()],
            d,
        );
        let mut logits = vec![0.0; t * v];
        matmul_bias(
            &norm_final.output,
            &self.parameters[self.layout.output_w.clone()],
            &self.parameters[self.layout.output_b.clone()],
            &mut logits,
            t,
            d,
            v,
        );
        if logits.iter().any(|x| !x.is_finite()) || !auxiliary_loss.is_finite() {
            return Err(ModelError("nonfinite forward computation"));
        }
        Ok(Cache {
            norm1,
            norm2,
            norm_final,
            qkv,
            attention_probs,
            context,
            router_probs,
            routes,
            accepted_mask,
            attempted,
            accepted,
            hidden,
            hidden_pre,
            expert_output,
            logits,
            auxiliary_loss,
            capacity,
        })
    }

    pub fn forward(&self, tokens: &[usize]) -> Result<Vec<f64>, ModelError> {
        Ok(self.forward_cached(tokens, false)?.logits)
    }

    pub fn routing_stats(&self, tokens: &[usize]) -> Result<RoutingStats, ModelError> {
        let cache = self.forward_cached(tokens, true)?;
        let mean_selected_gate = cache
            .routes
            .iter()
            .enumerate()
            .map(|(i, &expert)| cache.router_probs[i * self.config.experts + expert])
            .sum::<f64>()
            / tokens.len() as f64;
        Ok(RoutingStats {
            dropped: tokens.len() - cache.accepted.iter().sum::<usize>(),
            attempted: cache.attempted,
            accepted: cache.accepted,
            mean_selected_gate,
            capacity: cache.capacity,
        })
    }

    pub fn loss(&self, input: &[usize], targets: &[usize]) -> Result<f64, ModelError> {
        let cache = self.forward_cached(input, true)?;
        let (task_loss, _) = cross_entropy_with_gradient_from_logits(
            &cache.logits,
            targets,
            input.len(),
            self.config.vocab_size,
        )?;
        Ok(task_loss + cache.auxiliary_loss)
    }

    pub fn loss_and_gradient(
        &self,
        input: &[usize],
        targets: &[usize],
    ) -> Result<Gradients, ModelError> {
        let cache = self.forward_cached(input, true)?;
        let c = self.config;
        let (t, d, v) = (input.len(), c.width, c.vocab_size);
        let (task_loss, dlogits) =
            cross_entropy_with_gradient_from_logits(&cache.logits, targets, t, v)?;
        let mut grads = vec![0.0; self.parameters.len()];
        let mut dblock = vec![0.0; t * d];
        for i in 0..t {
            for j in 0..d {
                for k in 0..v {
                    grads[self.layout.output_w.start + j * v + k] +=
                        cache.norm_final.output[i * d + j] * dlogits[i * v + k];
                    dblock[i * d + j] += dlogits[i * v + k]
                        * self.parameters[self.layout.output_w.start + j * v + k];
                }
            }
            for k in 0..v {
                grads[self.layout.output_b.start + k] += dlogits[i * v + k];
            }
        }
        let dblock = layer_norm_backward(
            &cache.norm_final,
            &dblock,
            &self.parameters,
            &self.layout.norm_final,
            &mut grads,
            d,
        );
        let dnorm2 = self.moe_backward(&cache, &dblock, &mut grads);
        let mut dresidual = layer_norm_backward(
            &cache.norm2,
            &dnorm2,
            &self.parameters,
            &self.layout.norm2,
            &mut grads,
            d,
        );
        for (a, b) in dresidual.iter_mut().zip(&dblock) {
            *a += b;
        }
        let mut dnorm1 = vec![0.0; t * d];
        let mut dcontext = vec![0.0; t * d];
        for i in 0..t {
            for j in 0..d {
                for k in 0..d {
                    grads[self.layout.attention_w.start + j * d + k] +=
                        cache.context[i * d + j] * dresidual[i * d + k];
                    dcontext[i * d + j] += dresidual[i * d + k]
                        * self.parameters[self.layout.attention_w.start + j * d + k];
                }
            }
            for k in 0..d {
                grads[self.layout.attention_b.start + k] += dresidual[i * d + k];
            }
        }
        let scale = (d as f64).sqrt();
        let mut dqkv = vec![0.0; t * 3 * d];
        for i in 0..t {
            let mut dprob = vec![0.0; i + 1];
            for j in 0..=i {
                for k in 0..d {
                    dqkv[j * 3 * d + 2 * d + k] +=
                        cache.attention_probs[i * t + j] * dcontext[i * d + k];
                    dprob[j] += dcontext[i * d + k] * cache.qkv[j * 3 * d + 2 * d + k];
                }
            }
            let dot = (0..=i)
                .map(|j| dprob[j] * cache.attention_probs[i * t + j])
                .sum::<f64>();
            for j in 0..=i {
                let dscore = cache.attention_probs[i * t + j] * (dprob[j] - dot);
                for k in 0..d {
                    dqkv[i * 3 * d + k] += dscore * cache.qkv[j * 3 * d + d + k] / scale;
                    dqkv[j * 3 * d + d + k] += dscore * cache.qkv[i * 3 * d + k] / scale;
                }
            }
        }
        for i in 0..t {
            for j in 0..d {
                for k in 0..3 * d {
                    grads[self.layout.qkv_w.start + j * 3 * d + k] +=
                        cache.norm1.output[i * d + j] * dqkv[i * 3 * d + k];
                    dnorm1[i * d + j] += dqkv[i * 3 * d + k]
                        * self.parameters[self.layout.qkv_w.start + j * 3 * d + k];
                }
            }
            for k in 0..3 * d {
                grads[self.layout.qkv_b.start + k] += dqkv[i * 3 * d + k];
            }
        }
        let mut dx = layer_norm_backward(
            &cache.norm1,
            &dnorm1,
            &self.parameters,
            &self.layout.norm1,
            &mut grads,
            d,
        );
        for (a, b) in dx.iter_mut().zip(&dresidual) {
            *a += b;
        }
        for i in 0..t {
            for j in 0..d {
                grads[self.layout.token.start + input[i] * d + j] += dx[i * d + j];
                grads[self.layout.position.start + i * d + j] += dx[i * d + j];
            }
        }
        let loss = task_loss + cache.auxiliary_loss;
        if !loss.is_finite() || grads.iter().any(|x| !x.is_finite()) {
            return Err(ModelError("nonfinite loss or gradient"));
        }
        Ok(Gradients {
            loss,
            task_loss,
            auxiliary_loss: cache.auxiliary_loss,
            values: grads,
            routes: cache.routes,
            accepted: cache.accepted,
            dropped: t - cache.accepted_mask.iter().filter(|&&x| x).count(),
        })
    }

    pub fn apply_sgd(
        &mut self,
        gradients: &Gradients,
        learning_rate: f64,
    ) -> Result<(), ModelError> {
        if gradients.values.len() != self.parameters.len()
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
            || gradients.values.iter().any(|x| !x.is_finite())
        {
            return Err(ModelError("invalid gradient or learning rate"));
        }
        if self
            .parameters
            .iter()
            .zip(&gradients.values)
            .any(|(p, g)| !(*p - learning_rate * g).is_finite())
        {
            return Err(ModelError("SGD update would be nonfinite"));
        }
        for (parameter, gradient) in self.parameters.iter_mut().zip(&gradients.values) {
            *parameter -= learning_rate * gradient;
        }
        Ok(())
    }

    pub fn generate(
        &self,
        prompt: &[usize],
        new_tokens: usize,
        rng: &mut Rng,
        temperature: f64,
    ) -> Result<Vec<usize>, ModelError> {
        if prompt.is_empty() || !temperature.is_finite() || temperature <= 0.0 || new_tokens > 256 {
            return Err(ModelError("generation arguments are invalid"));
        }
        if prompt.iter().any(|&x| x >= self.config.vocab_size) {
            return Err(ModelError("prompt token exceeds vocabulary"));
        }
        let mut output = prompt.to_vec();
        for _ in 0..new_tokens {
            let start = output.len().saturating_sub(self.config.context);
            let logits = self.forward(&output[start..])?;
            let row = &logits[logits.len() - self.config.vocab_size..];
            output.push(sample(row, temperature, rng));
        }
        Ok(output)
    }
}

#[derive(Clone)]
struct NormCache {
    normalized: Vec<f64>,
    inverse_std: Vec<f64>,
    output: Vec<f64>,
}

fn layer_norm(x: &[f64], parameters: &[f64], width: usize) -> NormCache {
    let mut normalized = vec![0.0; x.len()];
    let mut output = vec![0.0; x.len()];
    let mut inverse_std = Vec::new();
    for (i, row) in x.chunks_exact(width).enumerate() {
        let mean = row.iter().sum::<f64>() / width as f64;
        let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / width as f64;
        let inverse = 1.0 / (variance + 1e-5).sqrt();
        inverse_std.push(inverse);
        for j in 0..width {
            let at = i * width + j;
            normalized[at] = (row[j] - mean) * inverse;
            output[at] = normalized[at] * parameters[j] + parameters[width + j];
        }
    }
    NormCache {
        normalized,
        inverse_std,
        output,
    }
}

fn layer_norm_backward(
    cache: &NormCache,
    dy: &[f64],
    parameters: &[f64],
    range: &Range<usize>,
    gradients: &mut [f64],
    width: usize,
) -> Vec<f64> {
    let mut dx = vec![0.0; dy.len()];
    for i in 0..cache.inverse_std.len() {
        let mut sum = 0.0;
        let mut dot = 0.0;
        for j in 0..width {
            let at = i * width + j;
            let dz = dy[at] * parameters[range.start + j];
            sum += dz;
            dot += dz * cache.normalized[at];
            gradients[range.start + j] += dy[at] * cache.normalized[at];
            gradients[range.start + width + j] += dy[at];
        }
        for j in 0..width {
            let at = i * width + j;
            dx[at] = cache.inverse_std[i]
                * (dy[at] * parameters[range.start + j]
                    - sum / width as f64
                    - cache.normalized[at] * dot / width as f64);
        }
    }
    dx
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (std::f64::consts::FRAC_2_PI.sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}
fn gelu_grad(x: f64) -> f64 {
    let a = std::f64::consts::FRAC_2_PI.sqrt();
    let u = (a * (x + 0.044715 * x.powi(3))).tanh();
    0.5 * (1.0 + u) + 0.5 * x * (1.0 - u * u) * a * (1.0 + 3.0 * 0.044715 * x * x)
}

fn matmul_bias(
    a: &[f64],
    b: &[f64],
    bias: &[f64],
    output: &mut [f64],
    rows: usize,
    inner: usize,
    cols: usize,
) {
    for i in 0..rows {
        for k in 0..cols {
            output[i * cols + k] = bias[k]
                + (0..inner)
                    .map(|j| a[i * inner + j] * b[j * cols + k])
                    .sum::<f64>();
        }
    }
}

fn softmax_in_place(values: &mut [f64]) {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

pub fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(index, _)| index)
}

fn cross_entropy_with_gradient_from_logits(
    logits: &[f64],
    targets: &[usize],
    rows: usize,
    vocab: usize,
) -> Result<(f64, Vec<f64>), ModelError> {
    if targets.len() != rows || targets.iter().any(|&target| target >= vocab) {
        return Err(ModelError("targets must match input length and vocabulary"));
    }
    let mut loss = 0.0;
    let mut grad = vec![0.0; logits.len()];
    for i in 0..rows {
        let row = &logits[i * vocab..(i + 1) * vocab];
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum = row.iter().map(|x| (x - max).exp()).sum::<f64>();
        loss += (max - row[targets[i]]) + sum.ln();
        for j in 0..vocab {
            grad[i * vocab + j] =
                ((row[j] - max).exp() / sum - f64::from(j == targets[i])) / rows as f64;
        }
    }
    if !loss.is_finite() || grad.iter().any(|x| !x.is_finite()) {
        return Err(ModelError("nonfinite cross-entropy"));
    }
    Ok((loss / rows as f64, grad))
}

fn sample(logits: &[f64], temperature: f64, rng: &mut Rng) -> usize {
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities: Vec<_> = logits.iter().map(|x| (x - maximum) / temperature).collect();
    softmax_in_place(&mut probabilities);
    let draw = rng.next_f64();
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if draw <= cumulative {
            return index;
        }
    }
    probabilities.len() - 1
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn state(self) -> u64 {
        self.0
    }
    pub fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / ((1_u64 << 53) as f64)
    }
}

#[derive(Clone, Debug)]
pub struct Trainer {
    pub model: Model,
    pub step: u64,
    pub cursor: usize,
    pub rng: Rng,
    training: Option<TrainingIdentity>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TrainingIdentity {
    fingerprint: u64,
    length: usize,
    sequence: usize,
    learning_rate: f64,
}
fn fingerprint(data: &[u8]) -> u64 {
    // FNV-1a detects accidental corpus changes; this is not a cryptographic authenticity check.
    data.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

impl Trainer {
    pub fn new(model: Model, seed: u64) -> Self {
        Self {
            model,
            step: 0,
            cursor: 0,
            rng: Rng::new(seed),
            training: None,
        }
    }

    pub fn train_step(
        &mut self,
        data: &[u8],
        sequence: usize,
        learning_rate: f64,
    ) -> Result<Gradients, ModelError> {
        if data.len() < 2 || sequence == 0 || sequence > self.model.config.context {
            return Err(ModelError("training data and sequence length are invalid"));
        }
        if self.cursor >= data.len() || self.step == u64::MAX {
            return Err(ModelError(
                "invalid training cursor or exhausted step counter",
            ));
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(ModelError("invalid learning rate"));
        }
        let identity = TrainingIdentity {
            fingerprint: fingerprint(data),
            length: data.len(),
            sequence,
            learning_rate,
        };
        if self.training.is_some_and(|saved| saved != identity) {
            return Err(ModelError(
                "resume requires identical corpus, sequence length, and learning rate",
            ));
        }
        let mut input = Vec::with_capacity(sequence);
        let mut targets = Vec::with_capacity(sequence);
        for offset in 0..sequence {
            input.push(data[(self.cursor + offset) % data.len()] as usize);
            targets.push(data[(self.cursor + offset + 1) % data.len()] as usize);
        }
        let gradients = self.model.loss_and_gradient(&input, &targets)?;
        self.model.apply_sgd(&gradients, learning_rate)?;
        self.training = Some(identity);
        self.cursor = (self.cursor + sequence) % data.len();
        self.step = self
            .step
            .checked_add(1)
            .ok_or(ModelError("step overflow"))?;
        Ok(gradients)
    }

    pub fn evaluate(&self, data: &[u8]) -> Result<f64, ModelError> {
        if data.len() < 2 {
            return Err(ModelError("evaluation needs at least two bytes"));
        }
        // Evaluation is uncapped and reports only next-token cross-entropy, averaged over all targets.
        let mut total = 0.0;
        for start in (0..data.len() - 1).step_by(self.model.config.context) {
            let length = (data.len() - 1 - start).min(self.model.config.context);
            let input: Vec<_> = data[start..start + length]
                .iter()
                .map(|&x| x as usize)
                .collect();
            let targets: Vec<_> = data[start + 1..=start + length]
                .iter()
                .map(|&x| x as usize)
                .collect();
            let logits = self.model.forward(&input)?;
            total += cross_entropy_with_gradient_from_logits(
                &logits,
                &targets,
                length,
                self.model.config.vocab_size,
            )?
            .0 * length as f64;
        }
        Ok(total / (data.len() - 1) as f64)
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        if self.model.parameters.iter().any(|x| !x.is_finite()) {
            return Err("cannot save nonfinite parameters".into());
        }
        self.model.config.validate()?;
        match self.training {
            None if self.step == 0 && self.cursor == 0 => {}
            Some(identity)
                if self.step > 0
                    && identity.length >= 2
                    && identity.sequence > 0
                    && identity.sequence <= self.model.config.context
                    && identity.learning_rate.is_finite()
                    && identity.learning_rate > 0.0
                    && self.cursor
                        == ((self.step as u128 * identity.sequence as u128)
                            % identity.length as u128) as usize => {}
            _ => return Err("inconsistent training state cannot be saved".into()),
        }
        let mut bytes = MAGIC.to_vec();
        let c = self.model.config;
        for value in [c.vocab_size, c.context, c.width, c.ff_width, c.experts] {
            bytes.extend_from_slice(&u64::try_from(value)?.to_le_bytes());
        }
        bytes.extend_from_slice(&c.capacity_factor.to_le_bytes());
        bytes.extend_from_slice(&c.balance_weight.to_le_bytes());
        bytes.extend_from_slice(&self.step.to_le_bytes());
        bytes.extend_from_slice(&u64::try_from(self.cursor)?.to_le_bytes());
        bytes.extend_from_slice(&self.rng.state().to_le_bytes());
        let identity = self.training.unwrap_or(TrainingIdentity {
            fingerprint: 0,
            length: 0,
            sequence: 0,
            learning_rate: 0.0,
        });
        for value in [
            identity.fingerprint,
            identity.length as u64,
            identity.sequence as u64,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&identity.learning_rate.to_le_bytes());
        bytes.extend_from_slice(&u64::try_from(self.model.parameters.len())?.to_le_bytes());
        for parameter in &self.model.parameters {
            bytes.extend_from_slice(&parameter.to_le_bytes());
        }
        let temporary = path.with_extension(format!("tmp-{}-{}", std::process::id(), self.step));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let result = (|| -> Result<(), Box<dyn Error>> {
            file.write_all(&bytes)?;
            file.sync_all()?;
            fs::rename(&temporary, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    pub fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        if fs::metadata(path)?.len() > 256 * 1024 * 1024 {
            return Err("checkpoint exceeds 256 MiB".into());
        }
        let mut bytes = Vec::new();
        fs::File::open(path)?
            .take(256 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 256 * 1024 * 1024 {
            return Err("checkpoint exceeds 256 MiB".into());
        }
        let mut reader = Reader {
            bytes: &bytes,
            at: 0,
        };
        if reader.take(8)? != MAGIC {
            return Err("bad checkpoint header".into());
        }
        let config = Config {
            vocab_size: reader.usize()?,
            context: reader.usize()?,
            width: reader.usize()?,
            ff_width: reader.usize()?,
            experts: reader.usize()?,
            capacity_factor: reader.f64()?,
            balance_weight: reader.f64()?,
        };
        config.validate()?;
        let step = reader.u64()?;
        let cursor = reader.usize()?;
        let rng_state = reader.u64()?;
        if rng_state == 0 {
            return Err("invalid zero RNG state".into());
        }
        let rng = Rng::new(rng_state);
        let identity = TrainingIdentity {
            fingerprint: reader.u64()?,
            length: reader.usize()?,
            sequence: reader.usize()?,
            learning_rate: reader.f64()?,
        };
        let training = if step == 0 {
            if cursor != 0
                || identity
                    != (TrainingIdentity {
                        fingerprint: 0,
                        length: 0,
                        sequence: 0,
                        learning_rate: 0.0,
                    })
            {
                return Err("invalid untrained checkpoint state".into());
            }
            None
        } else {
            if identity.length < 2
                || cursor >= identity.length
                || identity.sequence == 0
                || identity.sequence > config.context
                || !identity.learning_rate.is_finite()
                || identity.learning_rate <= 0.0
            {
                return Err("invalid training identity".into());
            }
            Some(identity)
        };
        if training.is_some_and(|identity| {
            cursor
                != ((step as u128 * identity.sequence as u128) % identity.length as u128) as usize
        }) {
            return Err("checkpoint cursor is inconsistent with step".into());
        }
        let count = reader.usize()?;
        let layout = Layout::new(config);
        if count != layout.total
            || bytes.len() - reader.at != count.checked_mul(8).ok_or("size overflow")?
        {
            return Err("checkpoint length does not match model shape".into());
        }
        let mut model = Model::new(config, 1)?;
        for parameter in &mut model.parameters {
            *parameter = reader.f64()?;
        }
        if model.parameters.iter().any(|x| !x.is_finite()) {
            return Err("checkpoint contains nonfinite parameters".into());
        }
        Ok(Self {
            model,
            step,
            cursor,
            rng,
            training,
        })
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], &'static str> {
        let end = self
            .at
            .checked_add(len)
            .ok_or("checkpoint offset overflow")?;
        let value = self.bytes.get(self.at..end).ok_or("truncated checkpoint")?;
        self.at = end;
        Ok(value)
    }
    fn u64(&mut self) -> Result<u64, &'static str> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn usize(&mut self) -> Result<usize, &'static str> {
        usize::try_from(self.u64()?).map_err(|_| "checkpoint dimension exceeds usize")
    }
    fn f64(&mut self) -> Result<f64, &'static str> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

pub fn serve_once(path: &Path, address: &str) -> Result<(), Box<dyn Error>> {
    let mut trainer = Trainer::load(path)?;
    let address: std::net::SocketAddr = address.parse()?;
    if !address.ip().is_loopback() {
        return Err("teaching server only binds loopback".into());
    }
    let listener = TcpListener::bind(address)?;
    println!("serving one request on http://{address}/generate?prompt=rust&tokens=24");
    listener.set_nonblocking(true)?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut stream = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(error)
                if error.kind() == std::io::ErrorKind::WouldBlock
                    && std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(std::time::Duration::from_millis(10))
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                return Err("no request arrived within 30 seconds".into())
            }
            Err(error) => return Err(error.into()),
        }
    };
    stream.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(5)))?;
    handle_connection(&mut trainer, &mut stream)
}

// Explicit bounded local HTTP evidence; no socket is opened by the default experiment.
pub fn serve_check(path: &Path) -> Result<(), Box<dyn Error>> {
    let mut trainer = Trainer::load(path)?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?;
    listener.set_nonblocking(true)?;
    let worker = std::thread::spawn(move || -> Result<(), String> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        && std::time::Instant::now() < deadline =>
                {
                    std::thread::sleep(std::time::Duration::from_millis(5))
                }
                Err(e) => return Err(e.to_string()),
            }
        };
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .map_err(|e| e.to_string())?;
        stream
            .set_write_timeout(Some(std::time::Duration::from_secs(5)))
            .map_err(|e| e.to_string())?;
        handle_connection(&mut trainer, &mut stream).map_err(|e| e.to_string())
    });
    let mut stream =
        std::net::TcpStream::connect_timeout(&address, std::time::Duration::from_secs(5))?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(5)))?;
    stream.write_all(b"GET /generate?prompt=rust&tokens=4 HTTP/1.1\r\nHost: localhost\r\n\r\n")?;
    let mut response = String::new();
    stream.take(8192).read_to_string(&mut response)?;
    worker.join().map_err(|_| "HTTP smoke worker panicked")??;
    if !response.starts_with("HTTP/1.1 200 OK\r\n") || !response.contains("{\"text\":") {
        return Err("HTTP smoke response was not valid generation".into());
    }
    println!("bounded loopback serving check passed: 200 OK, four requested new tokens, {} response bytes",response.len());
    Ok(())
}

fn handle_connection<S: Read + Write>(
    trainer: &mut Trainer,
    stream: &mut S,
) -> Result<(), Box<dyn Error>> {
    let response = (|| -> Result<String, Box<dyn Error>> {
        let mut bytes = Vec::new();
        // Read a complete bounded header: a TCP read need not contain a whole HTTP request.
        while !bytes.ends_with(b"\r\n\r\n") {
            if bytes.len() == 4096 {
                return Err("request header exceeds 4096 bytes".into());
            }
            let mut byte = [0u8; 1];
            stream.read_exact(&mut byte)?;
            bytes.push(byte[0]);
        }
        let (prompt, count) = parse_request(std::str::from_utf8(&bytes)?)?;
        let tokens: Vec<_> = prompt.bytes().map(usize::from).collect();
        let generated = trainer
            .model
            .generate(&tokens, count, &mut trainer.rng, 0.8)?;
        let text: String = generated
            .into_iter()
            .map(|token| char::from_u32(token as u32).unwrap_or('?'))
            .collect();
        Ok(format!("{{\"text\":\"{}\"}}", json_escape(&text)))
    })();
    let (status, body) = match response {
        Ok(body) => ("200 OK", body),
        Err(_) => (
            "400 Bad Request",
            "{\"error\":\"invalid or incomplete request\"}".into(),
        ),
    };
    write!(stream, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body)?;
    stream.flush()?;
    Ok(())
}

fn parse_request(request: &str) -> Result<(String, usize), &'static str> {
    if request.len() > 4096 || !request.ends_with("\r\n\r\n") {
        return Err("incomplete or oversized request");
    }
    let mut lines = request.split("\r\n");
    let first = lines.next().ok_or("empty request")?;
    let parts: Vec<_> = first.split(' ').collect();
    if parts.len() != 3 || parts[0] != "GET" || parts[2] != "HTTP/1.1" {
        return Err("expected HTTP/1.1 GET request");
    }
    let mut content_length = false;
    for line in lines.take_while(|line| !line.is_empty()) {
        let (key, value) = line.split_once(':').ok_or("malformed header")?;
        if key.is_empty() || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
            return Err("invalid header name");
        }
        if key.eq_ignore_ascii_case("transfer-encoding") {
            return Err("body transfer encoding is unsupported");
        }
        if key.eq_ignore_ascii_case("content-length") {
            if content_length || value.trim() != "0" {
                return Err("GET must have no body and one content length");
            }
            content_length = true;
        }
    }
    let query = parts[1]
        .strip_prefix("/generate?")
        .ok_or("unknown endpoint")?;
    let mut prompt = None;
    let mut tokens = None;
    for field in query.split('&') {
        let (key, value) = field.split_once('=').ok_or("malformed query")?;
        match key {
            "prompt" if prompt.is_none() => prompt = Some(percent_decode(value)?),
            "tokens" if tokens.is_none() => {
                tokens = Some(value.parse().map_err(|_| "invalid token count")?)
            }
            _ => return Err("unknown or duplicate query field"),
        }
    }
    let prompt = prompt.ok_or("prompt is required")?;
    let tokens = tokens.unwrap_or(24);
    if prompt.is_empty() || prompt.len() > 256 || !prompt.is_ascii() || tokens > 64 {
        return Err("prompt must be 1..256 ASCII bytes and tokens at most 64");
    }
    Ok((prompt, tokens))
}

fn percent_decode(value: &str) -> Result<String, &'static str> {
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => output.push(b' '),
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).map_err(|_| "bad escape")?;
                output.push(u8::from_str_radix(hex, 16).map_err(|_| "bad escape")?);
                i += 2;
            }
            b'%' => return Err("bad escape"),
            byte => output.push(byte),
        }
        i += 1;
    }
    String::from_utf8(output).map_err(|_| "prompt is not UTF-8")
}

fn json_escape(value: &str) -> String {
    value
        .chars()
        .flat_map(|c| match c {
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '"' => "\\\"".chars().collect(),
            '\n' => "\\n".chars().collect(),
            c if c.is_control() => "?".chars().collect(),
            c => vec![c],
        })
        .collect()
}

struct MoeCache {
    router_probs: Vec<f64>,
    routes: Vec<usize>,
    accepted_mask: Vec<bool>,
    attempted: Vec<usize>,
    accepted: Vec<usize>,
    hidden: Vec<f64>,
    hidden_pre: Vec<f64>,
    expert_output: Vec<f64>,
    block_output: Vec<f64>,
    auxiliary_loss: f64,
    capacity: usize,
}

// Original course-authored ASCII fixtures. The held-out sentence is never trained on.
const TRAIN: &[u8] = b"rust learns patterns. experts route tokens. rust predicts words. experts share work. rust learns words. experts route patterns. ";
const HELD_OUT: &[u8] = b"rust predicts patterns. experts learn words. ";

fn initialized(experts: usize) -> Result<Trainer, Box<dyn std::error::Error>> {
    let dense = Model::new(Config::tiny(1), 56)?;
    let mut model = Model::new(Config::tiny(experts), 56)?;
    // Match the entire shared trunk and expert 0 across the comparison, despite layout offsets.
    for (name, target) in model.parameter_spans() {
        if name.starts_with("router") {
            continue;
        }
        if let Some((_, source)) = dense
            .parameter_spans()
            .into_iter()
            .find(|(n, r)| n == &name && r.len() == target.len())
        {
            model.parameters_mut()[target].copy_from_slice(&dense.parameters()[source]);
        }
    }
    Ok(Trainer::new(model, 5600))
}

fn train(
    trainer: &mut Trainer,
    data: &[u8],
    steps: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if steps == 0 || steps > 100_000 {
        return Err("steps must be 1..=100000; extended training is explicit".into());
    }
    for _ in 0..steps {
        trainer.train_step(data, 16, 0.08)?;
    }
    Ok(())
}

fn demo() -> Result<(), Box<dyn std::error::Error>> {
    let mut dense = initialized(1)?;
    let mut moe = initialized(3)?;
    let dense_before = dense.evaluate(TRAIN)?;
    let moe_before = moe.evaluate(TRAIN)?;
    let parameters_before = moe.model.parameters().to_vec();
    train(&mut dense, TRAIN, 160)?;
    train(&mut moe, TRAIN, 160)?;
    let stats = moe
        .model
        .routing_stats(&TRAIN[..16].iter().map(|&x| x as usize).collect::<Vec<_>>())?;
    println!("same fixture: 160 SGD steps, 16 tokens/step, width 8, expert width 12; matched shared initialization");
    println!(
        "dense parameters: total {}, nonexpert+one-expert {}",
        dense.model.total_parameters(),
        dense.model.active_parameters_per_token()
    );
    println!(
        "configured three-expert parameters: total {}, nonexpert+one-expert {}",
        moe.model.total_parameters(),
        moe.model.active_parameters_per_token()
    );
    println!("uncapped task cross-entropy, mean over ALL corpus targets:");
    println!(
        "dense train loss {dense_before:.4} -> {:.4}; held-out {:.4}",
        dense.evaluate(TRAIN)?,
        dense.evaluate(HELD_OUT)?
    );
    println!(
        "configured three-expert train loss {moe_before:.4} -> {:.4}; held-out {:.4}",
        moe.evaluate(TRAIN)?,
        moe.evaluate(HELD_OUT)?
    );
    println!("configured training probe: routes {:?}; accepted {:?}; dropped {}; capacity {}; mean gate {:.3}", stats.attempted, stats.accepted, stats.dropped, stats.capacity, stats.mean_selected_gate);
    for (name, range) in moe.model.parameter_spans() {
        let change = parameters_before[range.clone()]
            .iter()
            .zip(&moe.model.parameters()[range])
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        println!("parameter change {name}: L2={change:.6}");
    }
    let generated = moe.model.generate(
        &b"rust".iter().map(|&x| x as usize).collect::<Vec<_>>(),
        32,
        &mut Rng::new(7),
        0.8,
    )?;
    println!(
        "sample: {:?}",
        generated
            .into_iter()
            .map(|x| char::from_u32(x as u32).unwrap_or('?'))
            .collect::<String>()
    );
    println!("Tiny overlapping-vocabulary fixture checks mechanics, not language quality, privacy, or scale.");
    Ok(())
}

fn corpus(path: Option<&String>, fallback: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let data = if let Some(path) = path {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(8 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        bytes
    } else {
        fallback.to_vec()
    };
    if data.len() < 2 || data.len() > 8 * 1024 * 1024 || !data.is_ascii() {
        return Err("corpus must be 2 bytes..8 MiB of licensed ASCII text".into());
    }
    Ok(data)
}

fn validate_split(train: &[u8], heldout: &[u8]) -> Result<(), &'static str> {
    if train == heldout {
        return Err("train and held-out text must differ");
    }
    let (short, long) = if train.len() < heldout.len() {
        (train, heldout)
    } else {
        (heldout, train)
    };
    // ponytail: borrowed 32-byte windows use O(short corpus length) memory; use an external dedup index for large corpora.
    let windows: std::collections::HashSet<&[u8]> = short.windows(32).collect();
    if long.windows(32).any(|window| windows.contains(window)) {
        return Err("train and held-out text share a 32-byte passage");
    }
    Ok(())
}

fn command(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::iter::once("56".to_owned())
        .chain(args.iter().cloned())
        .collect();
    match args.get(1).map(String::as_str) {
        None => demo(),
        Some(command @ ("train" | "resume")) if args.len() <= 6 => {
            let path = args.get(2).ok_or("usage: train|resume CHECKPOINT [STEPS] [TRAIN_TEXT] [HELDOUT_TEXT]")?;
            let steps = args.get(3).map_or(Ok(400), |x| x.parse::<usize>())?;
            if args.get(4).is_some() != args.get(5).is_some() { return Err("custom training text requires an explicit held-out text file".into()); }
            let data = corpus(args.get(4), TRAIN)?;
            let heldout = corpus(args.get(5), HELD_OUT)?;
            if args.get(4).is_some() { validate_split(&data, &heldout)?; }
            let mut trainer = if command == "resume" { Trainer::load(Path::new(path))? } else { initialized(3)? };
            train(&mut trainer, &data, steps)?;
            trainer.save(Path::new(path))?;
            println!("saved step {} checkpoint to {path}; train task CE {:.4}; held-out task CE {:.4}", trainer.step, trainer.evaluate(&data)?, trainer.evaluate(&heldout)?);
            Ok(())
        }
        Some("eval") if args.len() <= 4 => {
            let path = args.get(2).ok_or("usage: eval CHECKPOINT [HELDOUT_TEXT]")?;
            let data = corpus(args.get(3), HELD_OUT)?;
            let trainer = Trainer::load(Path::new(path))?;
            println!("checkpoint step {}; held-out task CE {:.6}; {} targets", trainer.step, trainer.evaluate(&data)?, data.len()-1);
            Ok(())
        }
        Some("generate") if args.len() <= 5 => {
            let path = args.get(2).ok_or("usage: generate CHECKPOINT PROMPT [TOKENS]")?;
            let prompt = args.get(3).ok_or("usage: generate CHECKPOINT PROMPT [TOKENS]")?;
            if !prompt.is_ascii() || prompt.is_empty() || prompt.len() > 256 { return Err("prompt must be 1..256 ASCII bytes".into()); }
            let count = args.get(4).map_or(Ok(32), |x| x.parse::<usize>())?;
            let trainer = Trainer::load(Path::new(path))?;
            let generated = trainer.model.generate(&prompt.bytes().map(usize::from).collect::<Vec<_>>(), count, &mut Rng::new(7), 0.8)?;
            println!("{}", generated.into_iter().map(|x| char::from_u32(x as u32).unwrap_or('?')).collect::<String>());
            Ok(())
        }
        Some("serve-check") if args.len() == 3 => serve_check(Path::new(&args[2])),
        Some("serve") if args.len() <= 4 => {
            let path = args.get(2).ok_or("usage: serve CHECKPOINT [ADDRESS]")?;
            serve_once(Path::new(path), args.get(3).map_or("127.0.0.1:8787", String::as_str))
        }
        _ => Err("commands: train|resume CHECKPOINT [STEPS] [TRAIN_TEXT] [HELDOUT_TEXT], eval CHECKPOINT [HELDOUT_TEXT], generate CHECKPOINT PROMPT [TOKENS], serve CHECKPOINT [ADDRESS], serve-check CHECKPOINT".into()),
    }
}

#[test]
fn split_guard_rejects_equal_or_shared_passages() {
    assert!(validate_split(b"same", b"same").is_err());
    assert!(validate_split(
        b"prefix abcdefghijklmnopqrstuvwxyz012345 suffix",
        b"abcdefghijklmnopqrstuvwxyz012345 different"
    )
    .is_err());
    assert!(validate_split(b"training words", b"held out words").is_ok());
}
