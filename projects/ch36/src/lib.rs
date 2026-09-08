//! A small, fully trainable decoder Transformer with explicit f32 backpropagation.

use std::{error::Error, fmt};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub vocab_size: usize,
    pub context: usize,
    pub width: usize,
    pub heads: usize,
    pub layers: usize,
    pub ff_width: usize,
}

impl Config {
    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,
            context: 16,
            width: 16,
            heads: 2,
            layers: 1,
            ff_width: 32,
        }
    }

    pub fn approximately_15m() -> Self {
        Self {
            vocab_size: 256,
            context: 128,
            width: 384,
            heads: 8,
            layers: 8,
            ff_width: 1536,
        }
    }

    pub fn parameter_count(self) -> Result<usize, ModelError> {
        self.validate()?;
        let d = self.width;
        let f = self.ff_width;
        let mul = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or(ModelError("parameter count overflow"))
        };
        let mut n = mul(self.vocab_size, d)?;
        n = add(n, mul(self.context, d)?)?;
        let mut per_layer = mul(3, mul(d, d)?)?;
        per_layer = add(per_layer, mul(3, d)?)?;
        per_layer = add(per_layer, mul(d, d)?)?;
        per_layer = add(per_layer, d)?;
        per_layer = add(per_layer, mul(4, d)?)?;
        per_layer = add(per_layer, mul(d, f)?)?;
        per_layer = add(per_layer, f)?;
        per_layer = add(per_layer, mul(f, d)?)?;
        per_layer = add(per_layer, d)?;
        n = add(n, mul(self.layers, per_layer)?)?;
        n = add(n, mul(2, d)?)?;
        n = add(n, mul(d, self.vocab_size)?)?;
        n = add(n, self.vocab_size)?;
        if n > isize::MAX as usize / std::mem::size_of::<f32>() {
            return Err(ModelError("parameter buffer exceeds Rust allocation limit"));
        }
        Ok(n)
    }

    fn validate(self) -> Result<(), ModelError> {
        if self.vocab_size < 2
            || self.context == 0
            || self.width == 0
            || self.heads == 0
            || self.layers == 0
            || self.ff_width == 0
        {
            return Err(ModelError(
                "all dimensions must be positive and vocabulary must contain at least two tokens",
            ));
        }
        if !self.width.is_multiple_of(self.heads) {
            return Err(ModelError("width must be divisible by heads"));
        }
        Ok(())
    }
}

fn add(a: usize, b: usize) -> Result<usize, ModelError> {
    a.checked_add(b)
        .ok_or(ModelError("parameter count overflow"))
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
struct LayerLayout {
    ln1_g: std::ops::Range<usize>,
    ln1_b: std::ops::Range<usize>,
    qkv_w: std::ops::Range<usize>,
    qkv_b: std::ops::Range<usize>,
    out_w: std::ops::Range<usize>,
    out_b: std::ops::Range<usize>,
    ln2_g: std::ops::Range<usize>,
    ln2_b: std::ops::Range<usize>,
    ff1_w: std::ops::Range<usize>,
    ff1_b: std::ops::Range<usize>,
    ff2_w: std::ops::Range<usize>,
    ff2_b: std::ops::Range<usize>,
}

#[derive(Clone, Debug)]
struct Layout {
    token: std::ops::Range<usize>,
    position: std::ops::Range<usize>,
    layers: Vec<LayerLayout>,
    final_g: std::ops::Range<usize>,
    final_b: std::ops::Range<usize>,
    lm_w: std::ops::Range<usize>,
    lm_b: std::ops::Range<usize>,
    total: usize,
}

fn take(cursor: &mut usize, len: usize) -> std::ops::Range<usize> {
    let start = *cursor;
    *cursor += len;
    start..*cursor
}

impl Layout {
    fn new(c: Config) -> Self {
        let (d, f, v) = (c.width, c.ff_width, c.vocab_size);
        let mut p = 0;
        let token = take(&mut p, v * d);
        let position = take(&mut p, c.context * d);
        let mut layers = Vec::with_capacity(c.layers);
        for _ in 0..c.layers {
            layers.push(LayerLayout {
                ln1_g: take(&mut p, d),
                ln1_b: take(&mut p, d),
                qkv_w: take(&mut p, d * 3 * d),
                qkv_b: take(&mut p, 3 * d),
                out_w: take(&mut p, d * d),
                out_b: take(&mut p, d),
                ln2_g: take(&mut p, d),
                ln2_b: take(&mut p, d),
                ff1_w: take(&mut p, d * f),
                ff1_b: take(&mut p, f),
                ff2_w: take(&mut p, f * d),
                ff2_b: take(&mut p, d),
            });
        }
        let final_g = take(&mut p, d);
        let final_b = take(&mut p, d);
        let lm_w = take(&mut p, d * v);
        let lm_b = take(&mut p, v);
        Self {
            token,
            position,
            layers,
            final_g,
            final_b,
            lm_w,
            lm_b,
            total: p,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Decoder {
    config: Config,
    layout: Layout,
    params: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Gradients {
    pub loss: f32,
    pub tokens: usize,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterSpan {
    pub name: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug)]
pub struct LayerTrace {
    pub queries: Vec<f32>,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub attention_probabilities: Vec<f32>,
}

#[derive(Clone)]
struct NormCache {
    x: Vec<f32>,
    mean: Vec<f32>,
    inv_std: Vec<f32>,
}

#[derive(Clone)]
struct LayerCache {
    n1: Vec<f32>,
    n1_cache: NormCache,
    qkv: Vec<f32>,
    probs: Vec<f32>,
    context: Vec<f32>,
    n2: Vec<f32>,
    n2_cache: NormCache,
    ff_pre: Vec<f32>,
    ff_act: Vec<f32>,
}

#[derive(Clone)]
struct ForwardCache {
    layers: Vec<LayerCache>,
    final_norm: Vec<f32>,
    final_cache: NormCache,
    logits: Vec<f32>,
}

trait MatMul {
    fn mm(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>>;
}

struct Cpu;
impl MatMul for Cpu {
    fn mm(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if a.len() != m * k || b.len() != k * n {
            return Err(ModelError("matrix shape mismatch").into());
        }
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for q in 0..k {
                let av = a[i * k + q];
                for j in 0..n {
                    out[i * n + j] += av * b[q * n + j];
                }
            }
        }
        Ok(out)
    }
}

#[cfg(feature = "gpu")]
struct GpuBackend<'a>(&'a ch30::Gpu);
#[cfg(feature = "gpu")]
impl MatMul for GpuBackend<'_> {
    fn mm(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        self.0.matmul(a, b, m, k, n)
    }
}

impl Decoder {
    pub fn new(config: Config, seed: u64) -> Result<Self, ModelError> {
        let checked_count = config.parameter_count()?;
        let layout = Layout::new(config);
        debug_assert_eq!(layout.total, checked_count);
        let mut rng = Rng::new(seed);
        let mut params = vec![0.0; layout.total];
        let scale = 0.02;
        for x in &mut params {
            *x = (rng.next_f32() * 2.0 - 1.0) * scale;
        }
        for layer in &layout.layers {
            params[layer.ln1_g.clone()].fill(1.0);
            params[layer.ln1_b.clone()].fill(0.0);
            params[layer.ln2_g.clone()].fill(1.0);
            params[layer.ln2_b.clone()].fill(0.0);
            params[layer.qkv_b.clone()].fill(0.0);
            params[layer.out_b.clone()].fill(0.0);
            params[layer.ff1_b.clone()].fill(0.0);
            params[layer.ff2_b.clone()].fill(0.0);
        }
        params[layout.final_g.clone()].fill(1.0);
        params[layout.final_b.clone()].fill(0.0);
        params[layout.lm_b.clone()].fill(0.0);
        Ok(Self {
            config,
            layout,
            params,
        })
    }

    pub fn config(&self) -> Config {
        self.config
    }
    pub fn parameter_count(&self) -> usize {
        self.params.len()
    }
    pub fn parameters(&self) -> &[f32] {
        &self.params
    }
    pub fn parameters_mut(&mut self) -> &mut [f32] {
        &mut self.params
    }

    pub fn parameter_spans(&self) -> Vec<ParameterSpan> {
        let mut spans = vec![
            span("token_embedding", &self.layout.token),
            span("position_embedding", &self.layout.position),
        ];
        for (i, l) in self.layout.layers.iter().enumerate() {
            for (suffix, range) in [
                ("ln1_gain", &l.ln1_g),
                ("ln1_bias", &l.ln1_b),
                ("qkv_weight", &l.qkv_w),
                ("qkv_bias", &l.qkv_b),
                ("attention_output_weight", &l.out_w),
                ("attention_output_bias", &l.out_b),
                ("ln2_gain", &l.ln2_g),
                ("ln2_bias", &l.ln2_b),
                ("ff1_weight", &l.ff1_w),
                ("ff1_bias", &l.ff1_b),
                ("ff2_weight", &l.ff2_w),
                ("ff2_bias", &l.ff2_b),
            ] {
                spans.push(span(&format!("layer.{i}.{suffix}"), range));
            }
        }
        spans.extend([
            span("final_norm_gain", &self.layout.final_g),
            span("final_norm_bias", &self.layout.final_b),
            span("output_weight", &self.layout.lm_w),
            span("output_bias", &self.layout.lm_b),
        ]);
        spans
    }

    pub fn trace(&self, tokens: &[usize]) -> Result<Vec<LayerTrace>, Box<dyn Error>> {
        let cache = self.forward_cached(tokens, &Cpu)?;
        let d = self.config.width;
        Ok(cache
            .layers
            .into_iter()
            .map(|layer| LayerTrace {
                queries: rows_part(&layer.qkv, tokens.len(), 3 * d, 0, d),
                keys: rows_part(&layer.qkv, tokens.len(), 3 * d, d, d),
                values: rows_part(&layer.qkv, tokens.len(), 3 * d, 2 * d, d),
                attention_probabilities: layer.probs,
            })
            .collect())
    }

    pub fn forward(&self, tokens: &[usize]) -> Result<Vec<f32>, Box<dyn Error>> {
        Ok(self.forward_cached(tokens, &Cpu)?.logits)
    }

    pub fn loss_and_grad(
        &self,
        input: &[usize],
        targets: &[usize],
    ) -> Result<Gradients, Box<dyn Error>> {
        self.loss_and_grad_backend(input, targets, &Cpu)
    }

    pub fn loss(&self, input: &[usize], targets: &[usize]) -> Result<f32, Box<dyn Error>> {
        self.validate_tokens(input)?;
        if input.len() != targets.len() || targets.iter().any(|&x| x >= self.config.vocab_size) {
            return Err(ModelError("targets must match input length and vocabulary").into());
        }
        let logits = self.forward_cached(input, &Cpu)?.logits;
        let loss = cross_entropy(&logits, targets, input.len(), self.config.vocab_size).0;
        if !loss.is_finite() {
            return Err(ModelError("nonfinite loss").into());
        }
        Ok(loss)
    }

    #[cfg(feature = "gpu")]
    pub fn forward_with_gpu(
        &self,
        tokens: &[usize],
        gpu: &ch30::Gpu,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        Ok(self.forward_cached(tokens, &GpuBackend(gpu))?.logits)
    }

    #[cfg(feature = "gpu")]
    pub fn loss_and_grad_with_gpu(
        &self,
        input: &[usize],
        targets: &[usize],
        gpu: &ch30::Gpu,
    ) -> Result<Gradients, Box<dyn Error>> {
        self.loss_and_grad_backend(input, targets, &GpuBackend(gpu))
    }

    pub fn apply_sgd(
        &mut self,
        gradients: &Gradients,
        learning_rate: f32,
    ) -> Result<(), ModelError> {
        if gradients.values.len() != self.params.len()
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
            || !gradients.loss.is_finite()
            || gradients.tokens == 0
            || gradients.values.iter().any(|g| !g.is_finite())
        {
            return Err(ModelError("invalid gradient or learning rate"));
        }
        if self
            .params
            .iter()
            .zip(&gradients.values)
            .any(|(p, g)| !(*p - learning_rate * g).is_finite())
        {
            return Err(ModelError("SGD update would produce a nonfinite parameter"));
        }
        for (p, g) in self.params.iter_mut().zip(&gradients.values) {
            *p -= learning_rate * g;
        }
        Ok(())
    }

    pub fn generate(
        &self,
        prompt: &[usize],
        new_tokens: usize,
        rng: &mut Rng,
        temperature: f32,
    ) -> Result<Vec<usize>, Box<dyn Error>> {
        if prompt.is_empty() || !temperature.is_finite() || temperature <= 0.0 {
            return Err(
                ModelError("generation needs a prompt and positive finite temperature").into(),
            );
        }
        let mut out = prompt.to_vec();
        for _ in 0..new_tokens {
            let start = out.len().saturating_sub(self.config.context);
            let logits = self.forward(&out[start..])?;
            let row = &logits[logits.len() - self.config.vocab_size..];
            out.push(sample_logits(row, temperature, rng));
        }
        Ok(out)
    }

    fn validate_tokens(&self, tokens: &[usize]) -> Result<(), ModelError> {
        if tokens.is_empty() || tokens.len() > self.config.context {
            return Err(ModelError("sequence length must be in 1..=context"));
        }
        if tokens.iter().any(|&x| x >= self.config.vocab_size) {
            return Err(ModelError("token id exceeds vocabulary"));
        }
        Ok(())
    }

    fn forward_cached(
        &self,
        tokens: &[usize],
        backend: &dyn MatMul,
    ) -> Result<ForwardCache, Box<dyn Error>> {
        self.validate_tokens(tokens)?;
        let (t, d, v) = (tokens.len(), self.config.width, self.config.vocab_size);
        let mut x = vec![0.0; t * d];
        for i in 0..t {
            for j in 0..d {
                x[i * d + j] = self.params[self.layout.token.start + tokens[i] * d + j]
                    + self.params[self.layout.position.start + i * d + j];
            }
        }
        let mut caches = Vec::with_capacity(self.config.layers);
        for l in &self.layout.layers {
            let (n1, n1_cache) = layer_norm(
                &x,
                t,
                d,
                &self.params[l.ln1_g.clone()],
                &self.params[l.ln1_b.clone()],
            );
            let mut qkv = backend.mm(&n1, &self.params[l.qkv_w.clone()], t, d, 3 * d)?;
            add_bias(&mut qkv, t, 3 * d, &self.params[l.qkv_b.clone()]);
            let (probs, context) = attention_forward(&qkv, t, d, self.config.heads);
            let mut attn = backend.mm(&context, &self.params[l.out_w.clone()], t, d, d)?;
            add_bias(&mut attn, t, d, &self.params[l.out_b.clone()]);
            let residual: Vec<f32> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
            let (n2, n2_cache) = layer_norm(
                &residual,
                t,
                d,
                &self.params[l.ln2_g.clone()],
                &self.params[l.ln2_b.clone()],
            );
            let mut ff_pre = backend.mm(
                &n2,
                &self.params[l.ff1_w.clone()],
                t,
                d,
                self.config.ff_width,
            )?;
            add_bias(
                &mut ff_pre,
                t,
                self.config.ff_width,
                &self.params[l.ff1_b.clone()],
            );
            let ff_act: Vec<f32> = ff_pre.iter().copied().map(gelu).collect();
            let mut ff = backend.mm(
                &ff_act,
                &self.params[l.ff2_w.clone()],
                t,
                self.config.ff_width,
                d,
            )?;
            add_bias(&mut ff, t, d, &self.params[l.ff2_b.clone()]);
            x = residual.iter().zip(&ff).map(|(a, b)| a + b).collect();
            caches.push(LayerCache {
                n1,
                n1_cache,
                qkv,
                probs,
                context,
                n2,
                n2_cache,
                ff_pre,
                ff_act,
            });
        }
        let (final_norm, final_cache) = layer_norm(
            &x,
            t,
            d,
            &self.params[self.layout.final_g.clone()],
            &self.params[self.layout.final_b.clone()],
        );
        let mut logits =
            backend.mm(&final_norm, &self.params[self.layout.lm_w.clone()], t, d, v)?;
        add_bias(&mut logits, t, v, &self.params[self.layout.lm_b.clone()]);
        if logits.iter().any(|x| !x.is_finite()) {
            return Err(ModelError("nonfinite decoder logits").into());
        }
        Ok(ForwardCache {
            layers: caches,
            final_norm,
            final_cache,
            logits,
        })
    }

    fn loss_and_grad_backend(
        &self,
        input: &[usize],
        targets: &[usize],
        backend: &dyn MatMul,
    ) -> Result<Gradients, Box<dyn Error>> {
        self.validate_tokens(input)?;
        if input.len() != targets.len() || targets.iter().any(|&x| x >= self.config.vocab_size) {
            return Err(ModelError("targets must match input length and vocabulary").into());
        }
        let cache = self.forward_cached(input, backend)?;
        let (t, d, v) = (input.len(), self.config.width, self.config.vocab_size);
        let (loss, mut dlogits) = cross_entropy(&cache.logits, targets, t, v);
        let mut grads = vec![0.0; self.params.len()];
        let final_t = transpose(&cache.final_norm, t, d);
        grads[self.layout.lm_w.clone()].copy_from_slice(&backend.mm(&final_t, &dlogits, d, t, v)?);
        sum_rows_into(&dlogits, t, v, &mut grads[self.layout.lm_b.clone()]);
        let lm_t = transpose(&self.params[self.layout.lm_w.clone()], d, v);
        let mut dx = backend.mm(&dlogits, &lm_t, t, v, d)?;
        let (next_dx, dg, db) = layer_norm_backward(
            &dx,
            &cache.final_cache,
            t,
            d,
            &self.params[self.layout.final_g.clone()],
        );
        dx = next_dx;
        grads[self.layout.final_g.clone()].copy_from_slice(&dg);
        grads[self.layout.final_b.clone()].copy_from_slice(&db);
        for (l, c) in self.layout.layers.iter().zip(&cache.layers).rev() {
            let mut dres = dx.clone();
            let dff = dx;
            let ff_act_t = transpose(&c.ff_act, t, self.config.ff_width);
            grads[l.ff2_w.clone()].copy_from_slice(&backend.mm(
                &ff_act_t,
                &dff,
                self.config.ff_width,
                t,
                d,
            )?);
            sum_rows_into(&dff, t, d, &mut grads[l.ff2_b.clone()]);
            let ff2_t = transpose(&self.params[l.ff2_w.clone()], self.config.ff_width, d);
            let mut dact = backend.mm(&dff, &ff2_t, t, d, self.config.ff_width)?;
            for (g, &z) in dact.iter_mut().zip(&c.ff_pre) {
                *g *= gelu_grad(z);
            }
            let n2_t = transpose(&c.n2, t, d);
            grads[l.ff1_w.clone()].copy_from_slice(&backend.mm(
                &n2_t,
                &dact,
                d,
                t,
                self.config.ff_width,
            )?);
            sum_rows_into(&dact, t, self.config.ff_width, &mut grads[l.ff1_b.clone()]);
            let ff1_t = transpose(&self.params[l.ff1_w.clone()], d, self.config.ff_width);
            let dn2 = backend.mm(&dact, &ff1_t, t, self.config.ff_width, d)?;
            let (from_n2, dg, db) =
                layer_norm_backward(&dn2, &c.n2_cache, t, d, &self.params[l.ln2_g.clone()]);
            grads[l.ln2_g.clone()].copy_from_slice(&dg);
            grads[l.ln2_b.clone()].copy_from_slice(&db);
            for (i, g) in from_n2.into_iter().enumerate() {
                dres[i] += g;
            }
            let mut dinput = dres.clone();
            let datt = dres;
            let ctx_t = transpose(&c.context, t, d);
            grads[l.out_w.clone()].copy_from_slice(&backend.mm(&ctx_t, &datt, d, t, d)?);
            sum_rows_into(&datt, t, d, &mut grads[l.out_b.clone()]);
            let out_t = transpose(&self.params[l.out_w.clone()], d, d);
            let dcontext = backend.mm(&datt, &out_t, t, d, d)?;
            let dqkv = attention_backward(&dcontext, &c.qkv, &c.probs, t, d, self.config.heads);
            let n1_t = transpose(&c.n1, t, d);
            grads[l.qkv_w.clone()].copy_from_slice(&backend.mm(&n1_t, &dqkv, d, t, 3 * d)?);
            sum_rows_into(&dqkv, t, 3 * d, &mut grads[l.qkv_b.clone()]);
            let qkv_t = transpose(&self.params[l.qkv_w.clone()], d, 3 * d);
            let dn1 = backend.mm(&dqkv, &qkv_t, t, 3 * d, d)?;
            let (from_n1, dg, db) =
                layer_norm_backward(&dn1, &c.n1_cache, t, d, &self.params[l.ln1_g.clone()]);
            grads[l.ln1_g.clone()].copy_from_slice(&dg);
            grads[l.ln1_b.clone()].copy_from_slice(&db);
            for (i, g) in from_n1.into_iter().enumerate() {
                dinput[i] += g;
            }
            dx = dinput;
        }
        for i in 0..t {
            for j in 0..d {
                grads[self.layout.token.start + input[i] * d + j] += dx[i * d + j];
                grads[self.layout.position.start + i * d + j] += dx[i * d + j];
            }
        }
        if !loss.is_finite() || grads.iter().any(|x| !x.is_finite()) {
            return Err(ModelError("nonfinite loss or gradient").into());
        }
        dlogits.clear();
        Ok(Gradients {
            loss,
            tokens: t,
            values: grads,
        })
    }
}

fn add_bias(x: &mut [f32], rows: usize, cols: usize, b: &[f32]) {
    for i in 0..rows {
        for j in 0..cols {
            x[i * cols + j] += b[j];
        }
    }
}

fn span(name: &str, range: &std::ops::Range<usize>) -> ParameterSpan {
    ParameterSpan {
        name: name.to_owned(),
        start: range.start,
        end: range.end,
    }
}

fn rows_part(x: &[f32], rows: usize, stride: usize, offset: usize, width: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * width);
    for row in 0..rows {
        out.extend_from_slice(&x[row * stride + offset..row * stride + offset + width]);
    }
    out
}
fn sum_rows_into(x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    out.fill(0.0);
    for i in 0..rows {
        for j in 0..cols {
            out[j] += x[i * cols + j];
        }
    }
}
fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    for i in 0..rows {
        for j in 0..cols {
            y[j * rows + i] = x[i * cols + j];
        }
    }
    y
}

fn layer_norm(x: &[f32], rows: usize, d: usize, g: &[f32], b: &[f32]) -> (Vec<f32>, NormCache) {
    let mut y = vec![0.0; x.len()];
    let mut mean = vec![0.0; rows];
    let mut inv_std = vec![0.0; rows];
    for i in 0..rows {
        let row = &x[i * d..(i + 1) * d];
        let m = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|z| (z - m) * (z - m)).sum::<f32>() / d as f32;
        let inv = (var + 1e-5).sqrt().recip();
        mean[i] = m;
        inv_std[i] = inv;
        for j in 0..d {
            y[i * d + j] = (row[j] - m) * inv * g[j] + b[j];
        }
    }
    (
        y,
        NormCache {
            x: x.to_vec(),
            mean,
            inv_std,
        },
    )
}

fn layer_norm_backward(
    dy: &[f32],
    c: &NormCache,
    rows: usize,
    d: usize,
    g: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dg = vec![0.0; d];
    let mut db = vec![0.0; d];
    let mut dx = vec![0.0; dy.len()];
    for i in 0..rows {
        let mut sum = 0.0;
        let mut sum_xhat = 0.0;
        for j in 0..d {
            let xhat = (c.x[i * d + j] - c.mean[i]) * c.inv_std[i];
            let q = dy[i * d + j] * g[j];
            sum += q;
            sum_xhat += q * xhat;
            dg[j] += dy[i * d + j] * xhat;
            db[j] += dy[i * d + j];
        }
        for j in 0..d {
            let xhat = (c.x[i * d + j] - c.mean[i]) * c.inv_std[i];
            let q = dy[i * d + j] * g[j];
            dx[i * d + j] = c.inv_std[i] * (d as f32 * q - sum - xhat * sum_xhat) / d as f32;
        }
    }
    (dx, dg, db)
}

fn attention_forward(qkv: &[f32], t: usize, d: usize, heads: usize) -> (Vec<f32>, Vec<f32>) {
    let dh = d / heads;
    let mut probs = vec![0.0; heads * t * t];
    let mut context = vec![0.0; t * d];
    let scale = (dh as f32).sqrt().recip();
    for h in 0..heads {
        for i in 0..t {
            let base = (h * t + i) * t;
            let mut max = f32::NEG_INFINITY;
            for j in 0..=i {
                let mut s = 0.0;
                for z in 0..dh {
                    s += qkv[i * 3 * d + h * dh + z] * qkv[j * 3 * d + d + h * dh + z];
                }
                let q = s * scale;
                probs[base + j] = q;
                max = max.max(q);
            }
            let mut sum = 0.0;
            for j in 0..=i {
                probs[base + j] = (probs[base + j] - max).exp();
                sum += probs[base + j];
            }
            for j in 0..=i {
                probs[base + j] /= sum;
                for z in 0..dh {
                    context[i * d + h * dh + z] +=
                        probs[base + j] * qkv[j * 3 * d + 2 * d + h * dh + z];
                }
            }
        }
    }
    (probs, context)
}

fn attention_backward(
    dc: &[f32],
    qkv: &[f32],
    p: &[f32],
    t: usize,
    d: usize,
    heads: usize,
) -> Vec<f32> {
    let dh = d / heads;
    let scale = (dh as f32).sqrt().recip();
    let mut dqkv = vec![0.0; qkv.len()];
    for h in 0..heads {
        for i in 0..t {
            let base = (h * t + i) * t;
            let mut dp = vec![0.0; i + 1];
            for j in 0..=i {
                for z in 0..dh {
                    let dcx = dc[i * d + h * dh + z];
                    dp[j] += dcx * qkv[j * 3 * d + 2 * d + h * dh + z];
                    dqkv[j * 3 * d + 2 * d + h * dh + z] += p[base + j] * dcx;
                }
            }
            let dot = (0..=i).map(|j| dp[j] * p[base + j]).sum::<f32>();
            for j in 0..=i {
                let ds = p[base + j] * (dp[j] - dot) * scale;
                for z in 0..dh {
                    let q = i * 3 * d + h * dh + z;
                    let k = j * 3 * d + d + h * dh + z;
                    dqkv[q] += ds * qkv[k];
                    dqkv[k] += ds * qkv[q];
                }
            }
        }
    }
    dqkv
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}
fn gelu_grad(x: f32) -> f32 {
    let u = 0.797_884_6 * (x + 0.044715 * x * x * x);
    0.5 * (1.0 + u.tanh())
        + 0.5 * x * (1.0 - u.tanh().powi(2)) * 0.797_884_6 * (1.0 + 3.0 * 0.044715 * x * x)
}

fn cross_entropy(logits: &[f32], targets: &[usize], rows: usize, v: usize) -> (f32, Vec<f32>) {
    let mut loss = 0.0;
    let mut grad = vec![0.0; logits.len()];
    for i in 0..rows {
        let row = &logits[i * v..(i + 1) * v];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = row.iter().map(|x| (*x - max).exp()).sum::<f32>();
        loss += sum.ln() + (max - row[targets[i]]);
        for j in 0..v {
            grad[i * v + j] = (row[j] - max).exp() / sum / rows as f32;
        }
        grad[i * v + targets[i]] -= 1.0 / rows as f32;
    }
    (loss / rows as f32, grad)
}

fn sample_logits(logits: &[f32], temperature: f32, rng: &mut Rng) -> usize {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = logits
        .iter()
        .map(|x| ((*x - max) / temperature).exp())
        .collect();
    let sum = weights.iter().sum::<f32>();
    let mut r = rng.next_f32() * sum;
    for (i, w) in weights.iter_mut().enumerate() {
        if r < *w {
            return i;
        }
        r -= *w;
    }
    weights.len() - 1
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng {
    state: u64,
}
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed },
        }
    }
    pub fn state(&self) -> u64 {
        self.state
    }
    pub fn from_state(state: u64) -> Result<Self, ModelError> {
        if state == 0 {
            Err(ModelError("RNG state cannot be zero"))
        } else {
            Ok(Self { state })
        }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cross_entropy_and_sampling_handle_large_shared_offsets() {
        let (loss, grad) = cross_entropy(&[1e8; 3], &[1], 1, 3);
        assert!((loss - 3.0_f32.ln()).abs() < 1e-6);
        assert!(grad.iter().sum::<f32>().abs() < 1e-6);
        let mut rng = Rng::new(1);
        for _ in 0..20 {
            assert_eq!(
                sample_logits(&[1e8, 1e8 + 8.0], f32::MIN_POSITIVE, &mut rng),
                1
            );
        }
        let mut model = Decoder::new(
            Config {
                vocab_size: 3,
                context: 2,
                width: 4,
                heads: 2,
                layers: 1,
                ff_width: 8,
            },
            1,
        )
        .unwrap();
        model.parameters_mut().fill(f32::MAX);
        assert!(model.forward(&[0]).is_err());
        assert!(model.loss(&[0], &[1]).is_err());
    }

    #[test]
    fn large_count_is_exact() {
        assert_eq!(
            Config::approximately_15m().parameter_count().unwrap(),
            14_442_496
        );
    }
    #[test]
    fn impossible_config_is_rejected_before_allocation() {
        assert!(Config {
            vocab_size: usize::MAX,
            context: 1,
            width: 2,
            heads: 1,
            layers: 1,
            ff_width: 2
        }
        .parameter_count()
        .is_err());
        assert!(Decoder::new(
            Config {
                vocab_size: 8,
                context: 2,
                width: 3,
                heads: 2,
                layers: 1,
                ff_width: 4
            },
            1
        )
        .is_err());
    }
    #[test]
    fn causal_mask_blocks_future_tokens() {
        let m = Decoder::new(
            Config {
                vocab_size: 8,
                context: 4,
                width: 4,
                heads: 2,
                layers: 1,
                ff_width: 8,
            },
            7,
        )
        .unwrap();
        let a = m.forward(&[1, 2, 3]).unwrap();
        let b = m.forward(&[1, 7, 6]).unwrap();
        for i in 0..8 {
            assert_eq!(a[i], b[i]);
        }
    }
    #[test]
    fn several_parameter_groups_pass_finite_differences() {
        let mut m = Decoder::new(
            Config {
                vocab_size: 7,
                context: 3,
                width: 4,
                heads: 2,
                layers: 1,
                ff_width: 6,
            },
            11,
        )
        .unwrap();
        let input = [1, 2, 3];
        let target = [2, 3, 4];
        let analytic = m.loss_and_grad(&input, &target).unwrap();
        let l = &m.layout;
        let indices = [
            l.token.start + 4,
            l.position.start + 2,
            l.layers[0].qkv_w.start + 3,
            l.layers[0].ln2_g.start,
            l.layers[0].ff1_w.start + 5,
            l.lm_w.start + 4,
        ];
        for &idx in &indices {
            let old = m.params[idx];
            let eps = 1e-3;
            m.params[idx] = old + eps;
            let plus = m.loss_and_grad(&input, &target).unwrap().loss;
            m.params[idx] = old - eps;
            let minus = m.loss_and_grad(&input, &target).unwrap().loss;
            m.params[idx] = old;
            let numeric = (plus - minus) / (2.0 * eps);
            let tol = 3e-3 + 3e-2 * numeric.abs();
            assert!(
                (analytic.values[idx] - numeric).abs() < tol,
                "idx {idx}: analytic={} numeric={numeric}",
                analytic.values[idx]
            );
        }
    }
    #[test]
    fn attention_and_both_layers_have_nontrivial_checked_gradients() {
        let config = Config {
            vocab_size: 9,
            context: 4,
            width: 4,
            heads: 2,
            layers: 2,
            ff_width: 7,
        };
        let mut model = Decoder::new(config, 123).unwrap();
        for (i, p) in model.params.iter_mut().enumerate() {
            *p += (i % 17) as f32 * 0.003 - 0.02;
        }
        let input = [1, 5, 3, 7];
        let targets = [5, 3, 7, 2];
        let analytic = model.loss_and_grad(&input, &targets).unwrap();
        let ranges = [
            model.layout.layers[0].qkv_w.clone(),
            model.layout.layers[0].ff1_w.clone(),
            model.layout.layers[1].qkv_w.clone(),
            model.layout.layers[1].ff2_w.clone(),
        ];
        for range in ranges {
            let idx = range
                .clone()
                .max_by(|&a, &b| {
                    analytic.values[a]
                        .abs()
                        .total_cmp(&analytic.values[b].abs())
                })
                .unwrap();
            assert!(
                analytic.values[idx].abs() > 1e-5,
                "gradient path was effectively zero at {idx}"
            );
            let old = model.params[idx];
            let eps = 1e-3;
            model.params[idx] = old + eps;
            let plus = model.loss_and_grad(&input, &targets).unwrap().loss;
            model.params[idx] = old - eps;
            let minus = model.loss_and_grad(&input, &targets).unwrap().loss;
            model.params[idx] = old;
            let numeric = (plus - minus) / (2.0 * eps);
            assert!((analytic.values[idx] - numeric).abs() < 4e-3 + 4e-2 * numeric.abs());
        }
    }
    #[test]
    fn layer_norm_backward_matches_nontrivial_finite_difference() {
        let x = vec![0.7, -1.2, 0.4, 2.0, -0.3, 0.8];
        let gain = [1.2, -0.7, 0.5];
        let bias = [0.1, 0.2, -0.1];
        let upstream = [0.4, -0.8, 0.3, 1.1, -0.2, 0.6];
        let (_, cache) = layer_norm(&x, 2, 3, &gain, &bias);
        let (dx, _, _) = layer_norm_backward(&upstream, &cache, 2, 3, &gain);
        for i in 0..x.len() {
            let mut xp = x.clone();
            let eps = 1e-3;
            xp[i] += eps;
            let plus = layer_norm(&xp, 2, 3, &gain, &bias)
                .0
                .iter()
                .zip(upstream)
                .map(|(a, b)| a * b)
                .sum::<f32>();
            xp[i] -= 2.0 * eps;
            let minus = layer_norm(&xp, 2, 3, &gain, &bias)
                .0
                .iter()
                .zip(upstream)
                .map(|(a, b)| a * b)
                .sum::<f32>();
            assert!((dx[i] - (plus - minus) / (2.0 * eps)).abs() < 2e-3);
        }
    }
    #[test]
    fn training_changes_representative_parameter_families_and_lowers_loss() {
        let mut m = Decoder::new(
            Config {
                vocab_size: 8,
                context: 4,
                width: 8,
                heads: 2,
                layers: 1,
                ff_width: 12,
            },
            19,
        )
        .unwrap();
        let before = m.clone();
        let x = [1, 2, 3, 1];
        let y = [2, 3, 1, 2];
        for _ in 0..40 {
            let g = m.loss_and_grad(&x, &y).unwrap();
            m.apply_sgd(&g, 0.08).unwrap();
        }
        assert!(m.loss_and_grad(&x, &y).unwrap().loss < before.loss_and_grad(&x, &y).unwrap().loss);
        let l = &m.layout;
        for r in [
            &l.token,
            &l.position,
            &l.layers[0].ln1_g,
            &l.layers[0].qkv_w,
            &l.layers[0].out_w,
            &l.layers[0].ln2_g,
            &l.layers[0].ff1_w,
            &l.layers[0].ff2_w,
            &l.final_g,
            &l.lm_w,
        ] {
            assert!(r.clone().any(|i| m.params[i] != before.params[i]));
        }
    }
    #[test]
    fn standalone_attention_backward_checks_q_k_and_v() {
        let (t, d, heads) = (3, 4, 2);
        let qkv = (0..t * 3 * d)
            .map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.11)
            .collect::<Vec<_>>();
        let (probabilities, _) = attention_forward(&qkv, t, d, heads);
        let upstream = (0..t * d)
            .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.17)
            .collect::<Vec<_>>();
        let analytic = attention_backward(&upstream, &qkv, &probabilities, t, d, heads);
        let eps = 1e-3;
        for part in 0..3 {
            let mut best = part * d;
            for row in 0..t {
                for col in part * d..(part + 1) * d {
                    let idx = row * 3 * d + col;
                    if analytic[idx].abs() > analytic[best].abs() {
                        best = idx;
                    }
                }
            }
            assert!(analytic[best].abs() > 2e-3);
            let mut plus = qkv.clone();
            plus[best] += eps;
            let lp = attention_forward(&plus, t, d, heads)
                .1
                .iter()
                .zip(&upstream)
                .map(|(a, b)| a * b)
                .sum::<f32>();
            let mut minus = qkv.clone();
            minus[best] -= eps;
            let lm = attention_forward(&minus, t, d, heads)
                .1
                .iter()
                .zip(&upstream)
                .map(|(a, b)| a * b)
                .sum::<f32>();
            let numeric = (lp - lm) / (2.0 * eps);
            assert!(
                (analytic[best] - numeric).abs() < 2e-4 + 3e-2 * numeric.abs(),
                "part {part}: {} != {numeric}",
                analytic[best]
            );
        }
    }
}
