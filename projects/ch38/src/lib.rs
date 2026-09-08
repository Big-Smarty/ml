use ch36::{Config, Decoder, Gradients, Rng};
use std::{
    error::Error,
    fs,
    io::{self, Read, Write},
    path::Path,
};

#[derive(Clone, Copy, Debug)]
pub struct TrainConfig {
    pub peak_learning_rate: f32,
    pub min_learning_rate: f32,
    pub warmup_steps: u64,
    pub total_steps: u64,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub clip_norm: f32,
}
impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            peak_learning_rate: 2e-3,
            min_learning_rate: 2e-4,
            warmup_steps: 5,
            total_steps: 100,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            clip_norm: 1.0,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Trainer {
    pub model: Decoder,
    pub config: TrainConfig,
    pub step: u64,
    pub data_cursor: usize,
    pub rng: Rng,
    pub data_fingerprint: u64,
    m: Vec<f32>,
    v: Vec<f32>,
}
impl Trainer {
    pub fn new(model: Decoder, config: TrainConfig, seed: u64) -> Result<Self, &'static str> {
        validate_train(config)?;
        let n = model.parameter_count();
        Ok(Self {
            model,
            config,
            step: 0,
            data_cursor: 0,
            rng: Rng::new(seed),
            data_fingerprint: 0,
            m: vec![0.0; n],
            v: vec![0.0; n],
        })
    }
    pub fn learning_rate(&self) -> f32 {
        schedule(self.config, self.step)
    }
    pub fn memory_bytes(&self) -> usize {
        (self.model.parameter_count() + self.m.len() + self.v.len()) * std::mem::size_of::<f32>()
    }
    pub fn train_accumulated(
        &mut self,
        data: &[usize],
        tokens: usize,
        micro_batches: usize,
    ) -> Result<f32, Box<dyn Error>> {
        self.accumulate(data, tokens, micro_batches, |model, x, y| {
            model.loss_and_gradient(x, y)
        })
    }
    #[cfg(feature = "gpu")]
    pub fn train_accumulated_gpu(
        &mut self,
        data: &[usize],
        tokens: usize,
        micro_batches: usize,
        gpu: &ch30::Gpu,
    ) -> Result<f32, Box<dyn Error>> {
        self.accumulate(data, tokens, micro_batches, |model, x, y| {
            model.loss_and_gradient_with_gpu(x, y, gpu)
        })
    }
    fn accumulate<F>(
        &mut self,
        data: &[usize],
        tokens: usize,
        micro_batches: usize,
        mut gradient: F,
    ) -> Result<f32, Box<dyn Error>>
    where
        F: FnMut(&Decoder, &[usize], &[usize]) -> Result<Gradients, Box<dyn Error>>,
    {
        self.validate_state()?;
        if tokens == 0
            || tokens > self.model.config().context
            || micro_batches == 0
            || data.len() < tokens.checked_add(1).ok_or("token count overflow")?
        {
            return Err("invalid accumulation dimensions or corpus too short".into());
        }
        let fingerprint = fingerprint(data);
        if self.data_fingerprint != 0 && self.data_fingerprint != fingerprint {
            return Err("training data differs from checkpoint".into());
        }
        let mut cursor = self.data_cursor;
        let mut rng = self.rng;
        let mut all = vec![0.0; self.model.parameter_count()];
        let mut loss = 0.0;
        for _ in 0..micro_batches {
            let max = data.len() - tokens;
            let start = cursor % max;
            let g = gradient(
                &self.model,
                &data[start..start + tokens],
                &data[start + 1..start + tokens + 1],
            )?;
            for (a, b) in all.iter_mut().zip(g.values) {
                *a += b / micro_batches as f32;
            }
            loss += g.loss / micro_batches as f32;
            cursor = cursor.checked_add(tokens).ok_or("data cursor overflow")? % max;
            let _ = rng.next_u64();
        }
        self.adamw(&mut all)?;
        self.data_cursor = cursor;
        self.rng = rng;
        self.data_fingerprint = fingerprint;
        Ok(loss)
    }
    fn adamw(&mut self, g: &mut [f32]) -> Result<(), Box<dyn Error>> {
        self.validate_state()?;
        if g.len() != self.model.parameter_count() {
            return Err("gradient shape is invalid".into());
        }
        clip_global_norm(g, self.config.clip_norm)?;
        let next_step = self.step.checked_add(1).ok_or("optimizer step overflow")?;
        let learning_rate = schedule(self.config, self.step);
        let b1t = 1.0 - self.config.beta1.powf(next_step as f32);
        let b2t = 1.0 - self.config.beta2.powf(next_step as f32);
        for (i, &gradient) in g.iter().enumerate() {
            let m = self.config.beta1 * self.m[i] + (1.0 - self.config.beta1) * gradient;
            let v = self.config.beta2 * self.v[i] + (1.0 - self.config.beta2) * gradient * gradient;
            let update = m / b1t / ((v / b2t).sqrt() + self.config.epsilon)
                + self.config.weight_decay * self.model.parameters()[i];
            let next = self.model.parameters()[i] - learning_rate * update;
            if !m.is_finite() || !v.is_finite() || v < 0.0 || !next.is_finite() {
                return Err("AdamW produced a nonfinite parameter".into());
            }
        }
        for (i, &gradient) in g.iter().enumerate() {
            self.m[i] = self.config.beta1 * self.m[i] + (1.0 - self.config.beta1) * gradient;
            self.v[i] =
                self.config.beta2 * self.v[i] + (1.0 - self.config.beta2) * gradient * gradient;
            let update = self.m[i] / b1t / ((self.v[i] / b2t).sqrt() + self.config.epsilon)
                + self.config.weight_decay * self.model.parameters()[i];
            self.model.parameters_mut()[i] -= learning_rate * update;
        }
        self.step = next_step;
        Ok(())
    }
    fn validate_state(&self) -> Result<(), Box<dyn Error>> {
        validate_train(self.config)?;
        let n = self.model.parameter_count();
        if self.m.len() != n
            || self.v.len() != n
            || self
                .model
                .parameters()
                .iter()
                .chain(&self.m)
                .chain(&self.v)
                .any(|x| !x.is_finite())
            || self.v.iter().any(|x| *x < 0.0)
        {
            return Err("trainer state is invalid".into());
        }
        Ok(())
    }
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        self.validate_state()?;
        let mut b = Vec::new();
        b.extend_from_slice(b"CH38LM02");
        let c = self.model.config();
        for x in [
            c.vocab_size,
            c.context,
            c.width,
            c.heads,
            c.layers,
            c.ff_width,
        ] {
            b.extend_from_slice(&u64::try_from(x)?.to_le_bytes());
        }
        for x in [
            self.config.peak_learning_rate,
            self.config.min_learning_rate,
            self.config.weight_decay,
            self.config.beta1,
            self.config.beta2,
            self.config.epsilon,
            self.config.clip_norm,
        ] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b.extend_from_slice(&self.config.warmup_steps.to_le_bytes());
        b.extend_from_slice(&self.config.total_steps.to_le_bytes());
        b.extend_from_slice(&self.step.to_le_bytes());
        b.extend_from_slice(&u64::try_from(self.data_cursor)?.to_le_bytes());
        b.extend_from_slice(&self.rng.state().to_le_bytes());
        b.extend_from_slice(&self.data_fingerprint.to_le_bytes());
        b.extend_from_slice(&u64::try_from(self.model.parameter_count())?.to_le_bytes());
        for values in [self.model.parameters(), &self.m, &self.v] {
            for x in values {
                b.extend_from_slice(&x.to_le_bytes());
            }
        }
        let tmp = path.with_extension(format!("tmp-{}-{}", std::process::id(), self.step));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        if let Err(error) = file.write_all(&b).and_then(|_| file.sync_all()) {
            let _ = fs::remove_file(&tmp);
            return Err(error.into());
        }
        drop(file);
        if let Err(error) = fs::rename(&tmp, path) {
            let _ = fs::remove_file(&tmp);
            return Err(error.into());
        }
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        if fs::metadata(path)?.len() > 1_073_741_824 {
            return Err("checkpoint exceeds 1 GiB safety limit".into());
        }
        let mut b = Vec::new();
        fs::File::open(path)?.read_to_end(&mut b)?;
        let mut r = Reader { b: &b, p: 0 };
        if r.bytes(8)? != b"CH38LM02" {
            return Err("bad checkpoint header".into());
        }
        let config = Config {
            vocab_size: r.usize()?,
            context: r.usize()?,
            width: r.usize()?,
            heads: r.usize()?,
            layers: r.usize()?,
            ff_width: r.usize()?,
        };
        let tc = TrainConfig {
            peak_learning_rate: r.f32()?,
            min_learning_rate: r.f32()?,
            weight_decay: r.f32()?,
            beta1: r.f32()?,
            beta2: r.f32()?,
            epsilon: r.f32()?,
            clip_norm: r.f32()?,
            warmup_steps: r.u64()?,
            total_steps: r.u64()?,
        };
        validate_train(tc)?;
        let step = r.u64()?;
        let data_cursor = r.usize()?;
        let rng = Rng::from_state(r.u64()?)?;
        let data_fingerprint = r.u64()?;
        let n = r.usize()?;
        if n != config.parameter_count()? {
            return Err("checkpoint parameter count does not match dimensions".into());
        }
        let remaining = n
            .checked_mul(3)
            .and_then(|x| x.checked_mul(4))
            .ok_or("checkpoint size overflow")?;
        if b.len().checked_sub(r.p) != Some(remaining) {
            return Err("checkpoint length does not match parameter count".into());
        }
        let mut model = Decoder::new(config, 1)?;
        let params = r.f32s(n)?;
        model.parameters_mut().copy_from_slice(&params);
        let m = r.f32s(n)?;
        let v = r.f32s(n)?;
        if r.p != b.len()
            || params.iter().chain(&m).chain(&v).any(|x| !x.is_finite())
            || v.iter().any(|x| *x < 0.0)
        {
            return Err("checkpoint length or values are invalid".into());
        }
        let trainer = Self {
            model,
            config: tc,
            step,
            data_cursor,
            rng,
            data_fingerprint,
            m,
            v,
        };
        trainer.validate_state()?;
        Ok(trainer)
    }
}
fn clip_global_norm(gradient: &mut [f32], limit: f32) -> Result<(), &'static str> {
    if !limit.is_finite() || limit <= 0.0 || gradient.iter().any(|value| !value.is_finite()) {
        return Err("gradient values or clipping limit are invalid");
    }
    let norm = gradient
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() {
        return Err("nonfinite gradient norm");
    }
    if norm > f64::from(limit) {
        let scale = (f64::from(limit) / norm) as f32;
        for value in gradient {
            *value *= scale;
        }
    }
    Ok(())
}
fn validate_train(c: TrainConfig) -> Result<(), &'static str> {
    if [
        c.peak_learning_rate,
        c.min_learning_rate,
        c.weight_decay,
        c.beta1,
        c.beta2,
        c.epsilon,
        c.clip_norm,
    ]
    .iter()
    .any(|x| !x.is_finite())
        || c.peak_learning_rate <= 0.0
        || c.min_learning_rate < 0.0
        || c.min_learning_rate > c.peak_learning_rate
        || c.weight_decay < 0.0
        || c.beta1 < 0.0
        || c.beta1 >= 1.0
        || c.beta2 < 0.0
        || c.beta2 >= 1.0
        || c.epsilon <= 0.0
        || c.clip_norm <= 0.0
        || c.total_steps == 0
        || c.warmup_steps > c.total_steps
    {
        return Err("invalid optimizer configuration");
    }
    Ok(())
}
fn schedule(c: TrainConfig, step: u64) -> f32 {
    if step >= c.total_steps {
        return c.min_learning_rate;
    }
    if c.warmup_steps > 0 && step < c.warmup_steps {
        return c.peak_learning_rate * (step as f32 + 1.0) / c.warmup_steps as f32;
    }
    let span = c.total_steps.saturating_sub(c.warmup_steps).max(1);
    let progress = step.saturating_sub(c.warmup_steps).min(span) as f32 / span as f32;
    c.min_learning_rate
        + 0.5
            * (c.peak_learning_rate - c.min_learning_rate)
            * (1.0 + (std::f32::consts::PI * progress).cos())
}
fn fingerprint(data: &[usize]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &x in data {
        for b in (x as u64).to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3)
        }
    }
    h
}
struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}
impl Reader<'_> {
    fn bytes(&mut self, n: usize) -> Result<&[u8], io::Error> {
        let end = self.p.checked_add(n).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "checkpoint offset overflow")
        })?;
        let x = self
            .b
            .get(self.p..end)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "truncated checkpoint"))?;
        self.p = end;
        Ok(x)
    }
    fn u64(&mut self) -> Result<u64, io::Error> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn usize(&mut self) -> Result<usize, Box<dyn Error>> {
        Ok(usize::try_from(self.u64()?)?)
    }
    fn f32(&mut self) -> Result<f32, io::Error> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, io::Error> {
        (0..n).map(|_| self.f32()).collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resume_reproduces_the_next_step() {
        let data = b"abcabcabcabc"
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();
        let model = Decoder::new(
            Config {
                vocab_size: 256,
                context: 4,
                width: 8,
                heads: 2,
                layers: 1,
                ff_width: 12,
            },
            38,
        )
        .unwrap();
        let mut a = Trainer::new(model, TrainConfig::default(), 9).unwrap();
        a.train_accumulated(&data, 4, 2).unwrap();
        let p = std::env::temp_dir().join(format!("ch38-{}.bin", std::process::id()));
        let tmp = p.with_extension(format!("tmp-{}-{}", std::process::id(), a.step));
        fs::write(&tmp, b"belongs to another writer").unwrap();
        assert!(a.save(&p).is_err());
        assert_eq!(fs::read(&tmp).unwrap(), b"belongs to another writer");
        fs::remove_file(&tmp).unwrap();
        a.save(&p).unwrap();
        let mut b = Trainer::load(&p).unwrap();
        let good = fs::read(&p).unwrap();
        let mut corrupt = good.clone();
        let end = corrupt.len();
        corrupt[end - 4..].copy_from_slice(&(-1.0_f32).to_le_bytes());
        fs::write(&p, &corrupt).unwrap();
        assert!(Trainer::load(&p).is_err());
        let mut trailing = good.clone();
        trailing.push(0);
        fs::write(&p, &trailing).unwrap();
        assert!(Trainer::load(&p).is_err());
        fs::write(&p, &good).unwrap();
        let la = a.train_accumulated(&data, 4, 2).unwrap();
        let lb = b.train_accumulated(&data, 4, 2).unwrap();
        assert_eq!(la.to_bits(), lb.to_bits());
        assert_eq!(a.model.parameters(), b.model.parameters());
        assert_eq!(a.data_cursor, b.data_cursor);
        assert_eq!(a.rng, b.rng);
        assert_eq!(a.step, b.step);
        assert_eq!(a.m, b.m);
        assert_eq!(a.v, b.v);
        assert_eq!(a.data_fingerprint, b.data_fingerprint);
        assert_eq!(a.learning_rate().to_bits(), b.learning_rate().to_bits());
        fs::remove_file(p).unwrap();
    }
    #[test]
    fn schedule_includes_first_warmup_and_decay_end() {
        let c = TrainConfig {
            peak_learning_rate: 1.0,
            min_learning_rate: 0.1,
            warmup_steps: 4,
            total_steps: 12,
            ..TrainConfig::default()
        };
        assert!((schedule(c, 0) - 0.25).abs() < 1e-7);
        assert!((schedule(c, 3) - 1.0).abs() < 1e-7);
        assert!((schedule(c, 12) - 0.1).abs() < 1e-7);
        let warmup_only = TrainConfig {
            warmup_steps: 12,
            ..c
        };
        assert_eq!(schedule(warmup_only, 11), 1.0);
        assert_eq!(schedule(warmup_only, 12), 0.1);
        assert_eq!(schedule(warmup_only, 100), 0.1);
    }
    #[test]
    fn first_adamw_step_matches_hand_calculation() {
        let mut model = Decoder::new(
            Config {
                vocab_size: 2,
                context: 1,
                width: 2,
                heads: 1,
                layers: 1,
                ff_width: 2,
            },
            1,
        )
        .unwrap();
        model.parameters_mut().fill(2.0);
        let config = TrainConfig {
            peak_learning_rate: 0.001,
            min_learning_rate: 0.001,
            warmup_steps: 0,
            total_steps: 1,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            clip_norm: 100.0,
        };
        let mut trainer = Trainer::new(model, config, 1).unwrap();
        let mut g = vec![0.2; trainer.model.parameter_count()];
        trainer.adamw(&mut g).unwrap();
        assert!((trainer.model.parameters()[0] - 1.99898).abs() < 2e-6);
        assert_eq!(trainer.step, 1);
    }
    #[test]
    fn clipping_uses_f64_norm_and_rejects_invalid_values() {
        let mut large = [f32::MAX, f32::MAX];
        clip_global_norm(&mut large, 1.0).unwrap();
        assert!((large[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((large[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!(clip_global_norm(&mut [f32::NAN], 1.0).is_err());
        assert!(clip_global_norm(&mut [1.0], 0.0).is_err());
    }
    #[test]
    fn accumulation_equals_explicit_average() {
        let data = b"abcdefghi".iter().map(|&x| x as usize).collect::<Vec<_>>();
        let model = Decoder::new(
            Config {
                vocab_size: 256,
                context: 4,
                width: 4,
                heads: 1,
                layers: 1,
                ff_width: 6,
            },
            8,
        )
        .unwrap();
        let mut accumulated = Trainer::new(model.clone(), TrainConfig::default(), 2).unwrap();
        let g1 = model.loss_and_gradient(&data[0..4], &data[1..5]).unwrap();
        let g2 = model.loss_and_gradient(&data[4..8], &data[5..9]).unwrap();
        let mut average = g1
            .values
            .iter()
            .zip(g2.values)
            .map(|(a, b)| (a + b) / 2.0)
            .collect::<Vec<_>>();
        let mut explicit = Trainer::new(model, TrainConfig::default(), 2).unwrap();
        explicit.adamw(&mut average).unwrap();
        let accumulated_loss = accumulated.train_accumulated(&data, 4, 2).unwrap();
        assert_eq!(
            accumulated_loss.to_bits(),
            ((g1.loss + g2.loss) / 2.0).to_bits()
        );
        assert_eq!(accumulated.model.parameters(), explicit.model.parameters());
    }
    #[test]
    fn malformed_checkpoint_is_rejected_before_allocation() {
        let p = std::env::temp_dir().join(format!("ch38-bad-{}.bin", std::process::id()));
        fs::write(&p, b"CH38LM02").unwrap();
        assert!(Trainer::load(&p).is_err());
        fs::remove_file(&p).unwrap();
    }
}
