//! Working baseline: full-prefix decoder with real sampling and request scheduling.
//! Incremental decoding for the Chapter 36 decoder, with real per-layer KV caches.
use ch36::{Config, Decoder, ParameterSpan};
use std::{collections::HashMap, error::Error, ops::Range};

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
    projection_rows: usize,
    tokens: Vec<usize>,
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
            projection_rows: 0,
            tokens: Vec::new(),
        }
    }
    fn parameter_slice(&self, name: &str) -> &[f32] {
        &self.model.parameters()[self.spans[name].clone()]
    }
    /// Cache one token at the next sequence position and return its [vocabulary] logits.
    /// This decoding step advances no optimizer state.
    fn step(&mut self, token_id: usize) -> Result<Vec<f32>, Box<dyn Error>> {
        // Stage 4: once --stages passes, replace this full-prefix body with
        // self.incremental_step(token_id). The orchestration is supplied below.
        let c = self.model.config();
        if token_id >= c.vocab_size || self.tokens.len() >= c.context {
            return Err("invalid token or full context".into());
        }
        self.tokens.push(token_id);
        let full = self.model.forward(&self.tokens)?;
        self.projection_rows += self.tokens.len() * c.layers;
        self.next_position += 1;
        Ok(full[full.len() - c.vocab_size..].to_vec())
    }
    fn incremental_step(&mut self, token_id: usize) -> Result<Vec<f32>, Box<dyn Error>> {
        let c = self.model.config();
        if self.model.parameters().iter().any(|p| !p.is_finite()) {
            return Err("decoder parameters must be finite".into());
        }
        if token_id >= c.vocab_size {
            return Err("token exceeds vocabulary".into());
        }
        if self.next_position >= c.context {
            return Err("cache reached model context; reset and replay a cropped window".into());
        }
        let mut x = self.embed_position(token_id);
        for layer in 0..c.layers {
            let q = self.project_append(layer, &x);
            let context = self.cached_attention(layer, &q);
            x = self.finish_block(layer, x, &context);
        }
        let logits = self.vocabulary_logits(&x);
        if logits.iter().any(|x| !x.is_finite()) {
            for layer in &mut self.layers {
                layer.keys.truncate(self.next_position * c.width);
                layer.values.truncate(self.next_position * c.width);
            }
            return Err("incremental decoder arithmetic produced nonfinite logits".into());
        }
        self.next_position += 1;
        Ok(logits)
    }
    /// Stage 1: one token [ID] and learned position -> residual stream [D].
    fn embed_position(&self, token_id: usize) -> Vec<f32> {
        let c = self.model.config();
        let mut x = vec![0.0; c.width];
        let tok = self.parameter_slice("token_embedding");
        let pos = self.parameter_slice("position_embedding");
        for j in 0..c.width {
            // LEARNER stage 1: choose the current learned position, not zero.
            x[j] = tok[token_id * c.width + j] + pos[j];
        }
        x
    }
    /// Stage 2: pre-normalize, project one row, persist K/V and return Q [D].
    fn project_append(&mut self, layer: usize, x: &[f32]) -> Vec<f32> {
        let c = self.model.config();
        let n1 = layer_norm(
            x,
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
        // LEARNER stage 2: retain earlier rows when adding the newest K/V.
        self.layers[layer].keys.clear();
        self.layers[layer].keys.extend_from_slice(k);
        self.layers[layer].values.clear();
        self.layers[layer].values.extend_from_slice(v);
        self.projection_rows += 1;
        q.to_vec()
    }
    /// Stage 3: new Q [D] attends over this layer's retained [positions,D] K/V.
    fn cached_attention(&self, layer: usize, q: &[f32]) -> Vec<f32> {
        let c = self.model.config();
        let positions = self.layers[layer].keys.len() / c.width;
        let mut context = vec![0.0; c.width];
        // LEARNER stage 3: replace latest-value copying with per-head scaled
        // dot products, stable softmax across positions, and weighted values.
        // Shapes and scalar helpers are supplied; the lesson gives the recurrence.
        let _query = q;
        context.copy_from_slice(&self.layers[layer].values[(positions - 1) * c.width..]);
        context
    }
    /// Supplied: attention output, both residuals and the GELU feed-forward branch.
    fn finish_block(&self, layer: usize, mut x: Vec<f32>, context: &[f32]) -> Vec<f32> {
        let c = self.model.config();
        let mut attn = matvec(
            context,
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
        x
    }
    /// Supplied: final normalization and vocabulary projection.
    fn vocabulary_logits(&self, x: &[f32]) -> Vec<f32> {
        let c = self.model.config();
        let n = layer_norm(
            x,
            self.parameter_slice("final_norm_gain"),
            self.parameter_slice("final_norm_bias"),
        );
        let mut logits = matvec(&n, self.parameter_slice("output_weight"), c.vocab_size);
        add(&mut logits, self.parameter_slice("output_bias"));
        logits
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
        || temperature <= 0.
        || top_k == 0
    {
        return Err("invalid sampling configuration");
    }
    // Baseline greedily chooses the maximum. LEARNER: retain top-k IDs, build
    // temperature-scaled mass and sample it with the supplied per-request RNG.
    let _draw = rng.next();
    Ok((1..logits.len()).fold(0, |best, i| if logits[i] > logits[best] { i } else { best }))
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
            .take(1)
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

fn experiment() -> Result<(), Box<dyn Error>> {
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
    println!("full-prefix baseline/oracle worst logit error: {worst:.8}");
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
        },
    )?;
    Ok(())
}

/// Run the useful baseline; --check separately verifies the learning goal.
pub fn run(args: &[String]) -> Result<(), String> {
    if args == ["--stages"] {
        return stage_diagnostics().map_err(|e| e.to_string());
    }
    if !args.is_empty() {
        return Err("unexpected experiment argument".into());
    }
    experiment().map_err(|e| e.to_string())
}

/// Compare every vocabulary coordinate and actual stored state on unseen prefixes.
pub fn check() -> Result<(), String> {
    let verify = || -> Result<(), Box<dyn Error>> {
        let model = tiny_model()?;
        let mut cached = CachedDecoder::new(&model);
        for (i, token) in [2, 9, 1, 6, 4, 3].iter().enumerate() {
            let tokens = [2, 9, 1, 6, 4, 3];
            let row = cached.step(*token)?;
            let oracle = model.forward(&tokens[..=i])?;
            if row.len() != model.config().vocab_size
                || max_error(&row, &oracle[oracle.len() - row.len()..]) > 2e-5
            {
                return Err(format!(
                    "GOAL_NOT_MET: prefix {}: cached logits differ from full decoder",
                    i + 1
                )
                .into());
            }
        }
        if cached.projection_rows != 6 * model.config().layers {
            return Err(format!(
                "GOAL_NOT_MET: recomputed {} layer-token projection rows; cache goal is {}",
                cached.projection_rows,
                6 * model.config().layers
            )
            .into());
        }
        let expected = 2 * model.config().layers * 6 * model.config().width;
        if cached.cached_values() != expected {
            return Err(format!("GOAL_NOT_MET: logits agree, but cache contains {} values; expected {expected}. Implement per-layer QKV append and cached attention", cached.cached_values()).into());
        }
        let events = run_request_batch(
            &model,
            &[("a", vec![1], 1), ("bb", vec![2, 3], 3), ("c", vec![5], 2)],
            |_, _| {},
        )?;
        if events.iter().map(|x| x.0.as_str()).collect::<Vec<_>>()
            != ["a", "bb", "c", "bb", "c", "bb"]
        {
            return Err(
                "GOAL_NOT_MET: scheduler must emit one token per active request per round".into(),
            );
        }
        let mut rng = SamplingRng(4);
        if sample_token(&[-9.0, 2.0, 1.0], 0.7, 1, &mut rng)? != 1 {
            return Err("GOAL_NOT_MET: top-k=1 must be greedy".into());
        }
        let mut seen = [0; 3];
        let mut rng = SamplingRng(40);
        for _ in 0..500 {
            seen[sample_token(&[2., 1., 0.], 2., 2, &mut rng)?] += 1;
        }
        if seen[0] == 0 || seen[1] == 0 || seen[2] != 0 {
            return Err("GOAL_NOT_MET: temperature/top-k sampling must draw both retained candidates and exclude the third".into());
        }
        println!("goal: every-prefix logits, {expected} cached scalars, top-k and three-request fairness pass");
        Ok(())
    };
    verify().map_err(|e| e.to_string())
}

/// Diagnostic fixtures return normally while reporting each subgoal independently.
fn stage_diagnostics() -> Result<(), Box<dyn Error>> {
    let model = tiny_model()?;
    let c = model.config();
    let mut cache = CachedDecoder::new(&model);
    cache.next_position = 2;
    let embedded = cache.embed_position(4);
    let expected: Vec<f32> = (0..c.width)
        .map(|j| {
            cache.parameter_slice("token_embedding")[4 * c.width + j]
                + cache.parameter_slice("position_embedding")[2 * c.width + j]
        })
        .collect();
    let error = max_error(&embedded, &expected);
    println!(
        "stage 1 embedding: error={error:.8}; {}",
        if error < 1e-7 { "PASS" } else { "GOAL_NOT_MET" }
    );
    cache.next_position = 0;
    cache.project_append(0, &vec![0.25; c.width]);
    let previous_keys = cache.layers[0].keys.clone();
    let previous_values = cache.layers[0].values.clone();
    cache.next_position = 1;
    cache.project_append(0, &vec![-0.5; c.width]);
    let retained = cache.layers[0].keys.len() == 2 * c.width
        && cache.layers[0].values.len() == 2 * c.width
        && cache.layers[0].keys[..c.width] == previous_keys
        && cache.layers[0].values[..c.width] == previous_values;
    println!(
        "stage 2 append: K/V rows={}/{}; earlier rows retained={retained}; {}",
        cache.layers[0].keys.len() / c.width,
        cache.layers[0].values.len() / c.width,
        if retained { "PASS" } else { "GOAL_NOT_MET" }
    );
    cache.layers[0].keys = vec![0.; 2 * c.width];
    cache.layers[0].values = [vec![1.; c.width], vec![3.; c.width]].concat();
    let context = cache.cached_attention(0, &vec![0.; c.width]);
    let error = max_error(&context, &vec![2.; c.width]);
    println!(
        "stage 3 attention: equal-score mean error={error:.8}; {}",
        if error < 1e-7 { "PASS" } else { "GOAL_NOT_MET" }
    );
    let mut cache = CachedDecoder::new(&model);
    let tokens = [2, 9, 1];
    let mut error = 0_f32;
    for end in 1..=tokens.len() {
        let row = cache.incremental_step(tokens[end - 1])?;
        let full = model.forward(&tokens[..end])?;
        error = error.max(max_error(&row, &full[full.len() - c.vocab_size..]));
    }
    let correct = error < 2e-5
        && cache.cached_values() == 2 * c.layers * 3 * c.width
        && cache.projection_rows == 3 * c.layers;
    println!(
        "stage 4 incremental prefix: error={error:.8}, scalars={}, projection rows={}; {}",
        cache.cached_values(),
        cache.projection_rows,
        if correct { "PASS" } else { "GOAL_NOT_MET" }
    );
    Ok(())
}

#[cfg(test)]
mod baseline_tests {
    #[test]
    fn supplied_baseline_runs() {
        assert!(super::run(&[]).is_ok());
    }
}
