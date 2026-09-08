//! A tiny contextual decoder whose dense feed-forward block is replaced by sparse experts.
//! The row-major layout and explicit backward pass continue the course-owned Chapter 36 model.
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
        let (t, d, f, e, v) = (tokens.len(), c.width, c.ff_width, c.experts, c.vocab_size);
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
        let mut router_probs = vec![0.0; t * e];
        let mut routes = vec![0; t];
        let mut attempted = vec![0; e];
        for i in 0..t {
            let row = &mut router_probs[i * e..(i + 1) * e];
            if e == 1 {
                row[0] = 1.0;
                attempted[0] += 1;
                continue;
            }
            for (expert, probability) in row.iter_mut().enumerate() {
                *probability = self.parameters[self.layout.router_b.start + expert]
                    + (0..d)
                        .map(|j| {
                            norm2.output[i * d + j]
                                * self.parameters[self.layout.router_w.start + j * e + expert]
                        })
                        .sum::<f64>();
            }
            softmax_in_place(row);
            routes[i] = argmax(row);
            attempted[routes[i]] += 1;
        }
        // ponytail: deterministic first-come capacity; batch-priority routing is the upgrade path.
        let capacity = if capped {
            capacity(t, e, c.capacity_factor)
        } else {
            t
        };
        let mut accepted = vec![0; e];
        let mut accepted_mask = vec![false; t];
        let mut hidden = vec![0.0; t * f];
        let mut hidden_pre = vec![0.0; t * f];
        let mut expert_output = vec![0.0; t * d];
        let mut block_output = residual.clone();
        for i in 0..t {
            let expert_id = routes[i];
            if accepted[expert_id] >= capacity {
                continue;
            }
            accepted[expert_id] += 1;
            accepted_mask[i] = true;
            let expert = &self.layout.experts[expert_id];
            for h in 0..f {
                let pre = self.parameters[expert.b1.start + h]
                    + (0..d)
                        .map(|j| {
                            norm2.output[i * d + j] * self.parameters[expert.w1.start + j * f + h]
                        })
                        .sum::<f64>();
                hidden_pre[i * f + h] = pre;
                hidden[i * f + h] = gelu(pre);
            }
            for j in 0..d {
                expert_output[i * d + j] = self.parameters[expert.b2.start + j]
                    + (0..f)
                        .map(|h| hidden[i * f + h] * self.parameters[expert.w2.start + h * d + j])
                        .sum::<f64>();
                block_output[i * d + j] +=
                    router_probs[i * e + expert_id] * expert_output[i * d + j];
            }
        }
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
        let auxiliary_loss = c.balance_weight
            * e as f64
            * attempted
                .iter()
                .enumerate()
                .map(|(expert, &count)| {
                    let frequency = count as f64 / t as f64;
                    let mean_probability =
                        (0..t).map(|i| router_probs[i * e + expert]).sum::<f64>() / t as f64;
                    frequency * mean_probability
                })
                .sum::<f64>();
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
        let (t, d, f, e, v) = (input.len(), c.width, c.ff_width, c.experts, c.vocab_size);
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
        let mut dnorm2 = vec![0.0; t * d];
        let mut drouter_prob = vec![0.0; t * e];
        for i in 0..t {
            if !cache.accepted_mask[i] {
                continue;
            }
            let expert_id = cache.routes[i];
            let expert = &self.layout.experts[expert_id];
            let gate = cache.router_probs[i * e + expert_id];
            let mut dhidden = vec![0.0; f];
            for j in 0..d {
                let dexpert = dblock[i * d + j] * gate;
                drouter_prob[i * e + expert_id] +=
                    dblock[i * d + j] * cache.expert_output[i * d + j];
                grads[expert.b2.start + j] += dexpert;
                for h in 0..f {
                    grads[expert.w2.start + h * d + j] += cache.hidden[i * f + h] * dexpert;
                    dhidden[h] += dexpert * self.parameters[expert.w2.start + h * d + j];
                }
            }
            for h in 0..f {
                let dpre = dhidden[h] * gelu_grad(cache.hidden_pre[i * f + h]);
                grads[expert.b1.start + h] += dpre;
                for j in 0..d {
                    grads[expert.w1.start + j * f + h] += cache.norm2.output[i * d + j] * dpre;
                    dnorm2[i * d + j] += dpre * self.parameters[expert.w1.start + j * f + h];
                }
            }
        }
        // Hard route frequencies are stop-gradient; mean probabilities remain differentiable.
        for i in 0..t {
            if e == 1 {
                continue;
            }
            for expert in 0..e {
                drouter_prob[i * e + expert] +=
                    c.balance_weight * e as f64 * cache.attempted[expert] as f64 / (t * t) as f64;
            }
            let dot = (0..e)
                .map(|expert| drouter_prob[i * e + expert] * cache.router_probs[i * e + expert])
                .sum::<f64>();
            for expert in 0..e {
                let dz = cache.router_probs[i * e + expert] * (drouter_prob[i * e + expert] - dot);
                grads[self.layout.router_b.start + expert] += dz;
                for j in 0..d {
                    grads[self.layout.router_w.start + j * e + expert] +=
                        cache.norm2.output[i * d + j] * dz;
                    dnorm2[i * d + j] +=
                        dz * self.parameters[self.layout.router_w.start + j * e + expert];
                }
            }
        }
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

fn argmax(values: &[f64]) -> usize {
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
    let (mut stream, _) = listener.accept()?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(5)))?;
    handle_connection(&mut trainer, &mut stream)
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

#[cfg(test)]
mod tests {
    use super::*;

    const DATA: &[u8] = b"rust routes tokens. rust learns words. ";

    fn span(model: &Model, name: &str) -> Range<usize> {
        model
            .parameter_spans()
            .into_iter()
            .find(|(candidate, _)| candidate == name)
            .unwrap()
            .1
    }

    #[test]
    fn one_expert_is_dense_feed_forward() {
        let model = Model::new(Config::tiny(1), 7).unwrap();
        let stats = model.routing_stats(&[1, 2, 3, 4]).unwrap();
        assert_eq!(stats.attempted, [4]);
        assert_eq!(stats.accepted, [4]);
        assert_eq!(stats.dropped, 0);
        assert_eq!(stats.mean_selected_gate, 1.0);
        assert_eq!(
            model.loss(&[1, 2], &[2, 3]).unwrap(),
            model.loss_and_gradient(&[1, 2], &[2, 3]).unwrap().task_loss
        );
    }

    #[test]
    fn router_and_expert_gradients_match_finite_differences() {
        let mut model = Model::new(
            Config {
                capacity_factor: 10.0,
                ..Config::tiny(3)
            },
            19,
        )
        .unwrap();
        let input = [7, 11, 13];
        let targets = [11, 13, 17];
        // Push expert 0 away from an argmax tie so the local finite difference keeps routes fixed.
        let router_bias = span(&model, "router_bias");
        model.parameters[router_bias.start] += 0.7;
        let analytic = model.loss_and_gradient(&input, &targets).unwrap();
        let expert = analytic.routes[0];
        let checks = [
            router_bias.start,
            span(&model, &format!("expert.{expert}.ff1_weight")).start,
        ];
        for index in checks {
            let original = model.parameters[index];
            let epsilon = 1e-5;
            model.parameters[index] = original + epsilon;
            let plus = model.loss(&input, &targets).unwrap();
            model.parameters[index] = original - epsilon;
            let minus = model.loss(&input, &targets).unwrap();
            model.parameters[index] = original;
            let numerical = (plus - minus) / (2.0 * epsilon);
            let tolerance = 1e-6 + 1e-4 * numerical.abs().max(analytic.values[index].abs());
            assert!(
                (numerical - analytic.values[index]).abs() <= tolerance,
                "index {index}: numerical {numerical}, analytic {}",
                analytic.values[index]
            );
        }
    }

    #[test]
    fn capacity_counts_attempts_and_drops() {
        let mut model = Model::new(
            Config {
                capacity_factor: 0.5,
                ..Config::tiny(2)
            },
            3,
        )
        .unwrap();
        let router_weight = span(&model, "router_weight");
        model.parameters[router_weight].fill(0.0);
        let router_bias = span(&model, "router_bias");
        model.parameters[router_bias.start] = 1.0;
        model.parameters[router_bias.start + 1] = 0.0;
        let stats = model.routing_stats(&[1, 2, 3, 4]).unwrap();
        assert_eq!(stats.capacity, 1);
        assert_eq!(stats.attempted, [4, 0]);
        assert_eq!(stats.accepted, [1, 0]);
        assert_eq!(stats.dropped, 3);
    }

    #[test]
    fn capacity_rounds_after_division() {
        assert_eq!(capacity(7, 3, 1.25), 3);
        assert_eq!(capacity(8, 3, 1.25), 4);
        assert_eq!(capacity(1, 8, 0.5), 1);
    }

    #[test]
    fn parameter_span_abi_is_stable() {
        let model = Model::new(
            Config {
                vocab_size: 3,
                context: 2,
                width: 2,
                ff_width: 3,
                experts: 2,
                capacity_factor: 1.25,
                balance_weight: 0.01,
            },
            1,
        )
        .unwrap();
        let expected = [
            ("norm1", 10, 14),
            ("norm2", 14, 18),
            ("norm_final", 18, 22),
            ("token_embedding", 0, 6),
            ("position_embedding", 6, 10),
            ("qkv_weight", 22, 34),
            ("qkv_bias", 34, 40),
            ("attention_output_weight", 40, 44),
            ("attention_output_bias", 44, 46),
            ("router_weight", 46, 50),
            ("router_bias", 50, 52),
            ("expert.0.ff1_weight", 52, 58),
            ("expert.0.ff1_bias", 58, 61),
            ("expert.0.ff2_weight", 61, 67),
            ("expert.0.ff2_bias", 67, 69),
            ("expert.1.ff1_weight", 69, 75),
            ("expert.1.ff1_bias", 75, 78),
            ("expert.1.ff2_weight", 78, 84),
            ("expert.1.ff2_bias", 84, 86),
            ("output_weight", 86, 92),
            ("output_bias", 92, 95),
        ];
        let spans = model.parameter_spans();
        assert_eq!(spans.len(), expected.len());
        for ((name, range), &(expected_name, start, end)) in spans.iter().zip(&expected) {
            assert_eq!(
                (name.as_str(), range.start, range.end),
                (expected_name, start, end)
            );
        }
        assert_eq!(model.total_parameters(), 95);
        assert_eq!(model.active_parameters_per_token(), 78);
        let tiny = Model::new(Config::tiny(3), 1).unwrap();
        assert_eq!(tiny.total_parameters(), 3_303);
        assert_eq!(tiny.active_parameters_per_token(), 2_879);
    }

    #[test]
    fn checkpoint_resume_is_exact() {
        let mut original = Trainer::new(Model::new(Config::tiny(3), 8).unwrap(), 9);
        for _ in 0..3 {
            original.train_step(DATA, 8, 0.05).unwrap();
        }
        let path = std::env::temp_dir().join(format!("ch56-resume-{}.bin", std::process::id()));
        original.save(&path).unwrap();
        let mut restored = Trainer::load(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(original.model.parameters, restored.model.parameters);
        assert_eq!(
            (original.step, original.cursor, original.rng),
            (restored.step, restored.cursor, restored.rng)
        );
        original.train_step(DATA, 8, 0.05).unwrap();
        restored.train_step(DATA, 8, 0.05).unwrap();
        assert_eq!(original.model.parameters, restored.model.parameters);
    }

    #[test]
    fn training_changes_trunk_router_and_expert() {
        let mut trainer = Trainer::new(
            Model::new(
                Config {
                    capacity_factor: 10.0,
                    ..Config::tiny(3)
                },
                5,
            )
            .unwrap(),
            6,
        );
        let before = trainer.model.parameters.clone();
        let gradients = trainer.train_step(DATA, 12, 0.05).unwrap();
        for name in [
            "token_embedding",
            "position_embedding",
            "qkv_weight",
            "attention_output_weight",
            "norm1",
            "norm2",
            "norm_final",
            "output_weight",
            "router_weight",
        ] {
            let range = span(&trainer.model, name);
            assert_ne!(&before[range.clone()], &trainer.model.parameters[range]);
        }
        let selected = gradients.routes[0];
        let range = span(&trainer.model, &format!("expert.{selected}.ff1_weight"));
        assert_ne!(&before[range.clone()], &trainer.model.parameters[range]);
    }

    #[test]
    fn request_boundary_is_strict() {
        assert_eq!(
            parse_request("GET /generate?prompt=rust+moe&tokens=4 HTTP/1.1\r\n\r\n").unwrap(),
            ("rust moe".into(), 4)
        );
        assert!(parse_request("POST /generate?prompt=x HTTP/1.1\r\n\r\n").is_err());
        assert!(parse_request("GET /generate?prompt=x&tokens=999 HTTP/1.1\r\n\r\n").is_err());
    }
}

#[cfg(test)]
mod verification;
