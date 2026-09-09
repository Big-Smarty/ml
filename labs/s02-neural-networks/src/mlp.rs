//! Supplied MLP cache, validated model construction, and checkpoint I/O.
use crate::data::{Dataset, CLASSES};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};
const MAGIC: &[u8; 8] = b"MLCH11\0\0";
#[derive(Clone, Debug)]
pub struct Mlp {
    pub input: usize,
    pub hidden: usize,
    pub parameters: Vec<f64>,
}
impl Mlp {
    pub fn checked_count(input: usize, hidden: usize) -> Option<usize> {
        hidden
            .checked_mul(input)?
            .checked_add(hidden)?
            .checked_add(CLASSES.checked_mul(hidden)?)?
            .checked_add(CLASSES)
    }
    pub fn new(input: usize, hidden: usize) -> Result<Self, String> {
        if input == 0 || hidden == 0 {
            return Err("model dimensions must be positive".into());
        }
        let mut seed = 7u64;
        let mut parameters =
            vec![0.; Self::checked_count(input, hidden).ok_or("model dimensions overflow")?];
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
        Ok(Self {
            input,
            hidden,
            parameters,
        })
    }
    pub fn parameter_offsets(&self) -> (usize, usize, usize, usize) {
        let weights1 = 0;
        let bias1 = self.hidden * self.input;
        let weights2 = bias1 + self.hidden;
        let bias2 = weights2 + CLASSES * self.hidden;
        (weights1, bias1, weights2, bias2)
    }
    pub fn forward(&self, image: &[f64]) -> (Vec<f64>, [f64; CLASSES]) {
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
    pub fn probabilities(logits: [f64; CLASSES]) -> [f64; CLASSES] {
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut probabilities = logits.map(|logit| (logit - maximum).exp());
        let sum = probabilities.iter().sum::<f64>();
        for probability in &mut probabilities {
            *probability /= sum
        }
        probabilities
    }
    pub fn cross_entropy_from_logits(logits: [f64; CLASSES], target: usize) -> f64 {
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (maximum - logits[target])
            + logits
                .iter()
                .map(|logit| (logit - maximum).exp())
                .sum::<f64>()
                .ln()
    }
    pub fn metrics(&self, data: &Dataset) -> Result<(f64, f64), String> {
        if data.in_features() != self.input || data.len() == 0 {
            return Err("evaluation shape mismatch".into());
        }
        let (mut loss, mut good) = (0., 0usize);
        for n in 0..data.len() {
            let (_, logits) = self.forward(data.image(n));
            loss += Self::cross_entropy_from_logits(logits, data.labels[n] as usize);
            let prediction = crate::linear::predict(logits);
            good += (prediction == data.labels[n] as usize) as usize
        }
        if !loss.is_finite() {
            return Err("evaluation loss overflowed".into());
        }
        Ok((loss / data.len() as f64, good as f64 / data.len() as f64))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Momentum,
    Adam,
}
#[derive(Clone, Debug)]
pub struct Optimizer {
    pub kind: Kind,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub step_count: u64,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}
impl Optimizer {
    pub fn new(kind: Kind, n: usize) -> Self {
        let learning_rate = if kind == Kind::Adam { 0.01 } else { 0.1 };
        Self {
            kind,
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            step_count: 0,
            m: vec![0.; n],
            v: vec![0.; n],
        }
    }
}
pub fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend(v.to_le_bytes())
}
pub fn put_f64(b: &mut Vec<u8>, v: f64) {
    b.extend(v.to_le_bytes())
}
pub fn save(path: &Path, model: &Mlp, optimizer: &Optimizer) -> Result<(), String> {
    let mut b = MAGIC.to_vec();
    b.extend(1u32.to_le_bytes());
    put_u64(&mut b, model.input as u64);
    put_u64(&mut b, model.hidden as u64);
    put_u64(&mut b, model.parameters.len() as u64);
    b.push(if optimizer.kind == Kind::Adam { 1 } else { 0 });
    put_u64(&mut b, optimizer.step_count);
    for v in [
        optimizer.learning_rate,
        optimizer.beta1,
        optimizer.beta2,
        optimizer.eps,
    ] {
        put_f64(&mut b, v)
    }
    put_u64(&mut b, optimizer.m.len() as u64);
    put_u64(&mut b, optimizer.v.len() as u64);
    for a in [&model.parameters, &optimizer.m, &optimizer.v] {
        for &v in a {
            put_f64(&mut b, v)
        }
    }
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let mut f = fs::File::create_new(&tmp)
        .map_err(|e| format!("cannot create checkpoint temp file {}: {e}", tmp.display()))?;
    f.write_all(&b)
        .and_then(|_| f.sync_all())
        .map_err(|e| format!("cannot write checkpoint: {e}"))?;
    fs::rename(&tmp, path).map_err(|e| format!("cannot install checkpoint: {e}"))
}
pub fn restore(path: &Path) -> Result<(Mlp, Optimizer), String> {
    let b = fs::read(path).map_err(|e| format!("cannot read checkpoint: {e}"))?;
    let mut p = 0usize;
    pub fn take<const N: usize>(b: &[u8], p: &mut usize) -> Result<[u8; N], String> {
        let end = p.checked_add(N).ok_or("checkpoint offset overflow")?;
        let s = b.get(*p..end).ok_or("truncated checkpoint")?;
        *p = end;
        Ok(s.try_into().map_err(|_| "invalid checkpoint field")?)
    }
    if take::<8>(&b, &mut p)? != *MAGIC {
        return Err("checkpoint magic mismatch".into());
    }
    if u32::from_le_bytes(take(&b, &mut p)?) != 1 {
        return Err("unsupported checkpoint version".into());
    }
    let as_usize = |v: u64| {
        usize::try_from(v).map_err(|_| "checkpoint length exceeds this platform".to_string())
    };
    let input = as_usize(u64::from_le_bytes(take(&b, &mut p)?))?;
    let hidden = as_usize(u64::from_le_bytes(take(&b, &mut p)?))?;
    let n = as_usize(u64::from_le_bytes(take(&b, &mut p)?))?;
    let kind = match take::<1>(&b, &mut p)?[0] {
        0 => Kind::Momentum,
        1 => Kind::Adam,
        _ => return Err("unknown optimizer kind".into()),
    };
    let step_count = u64::from_le_bytes(take(&b, &mut p)?);
    let learning_rate = f64::from_le_bytes(take(&b, &mut p)?);
    let beta1 = f64::from_le_bytes(take(&b, &mut p)?);
    let beta2 = f64::from_le_bytes(take(&b, &mut p)?);
    let eps = f64::from_le_bytes(take(&b, &mut p)?);
    let nm = as_usize(u64::from_le_bytes(take(&b, &mut p)?))?;
    let nv = as_usize(u64::from_le_bytes(take(&b, &mut p)?))?;
    let expected =
        Mlp::checked_count(input, hidden).ok_or("checkpoint model dimensions overflow")?;
    let values = n
        .checked_add(nm)
        .and_then(|v| v.checked_add(nv))
        .and_then(|v| v.checked_mul(8))
        .ok_or("checkpoint state size overflow")?;
    if input == 0
        || hidden == 0
        || n != expected
        || nm != n
        || nv != n
        || b.len().checked_sub(p) != Some(values)
    {
        return Err("checkpoint dimensions, state lengths, or byte length are invalid".into());
    }
    if !learning_rate.is_finite()
        || learning_rate <= 0.
        || !eps.is_finite()
        || eps <= 0.
        || ![beta1, beta2]
            .iter()
            .all(|v| v.is_finite() && *v >= 0. && *v < 1.)
    {
        return Err("checkpoint optimizer hyperparameters are invalid".into());
    }
    let mut read_vec = |len| -> Result<Vec<f64>, String> {
        (0..len)
            .map(|_| Ok(f64::from_le_bytes(take(&b, &mut p)?)))
            .collect()
    };
    let parameters = read_vec(n)?;
    let ms = read_vec(n)?;
    let vs = read_vec(n)?;
    if parameters
        .iter()
        .chain(&ms)
        .chain(&vs)
        .any(|v| !v.is_finite())
        || (kind == Kind::Adam && vs.iter().any(|&v| v < 0.))
    {
        return Err("checkpoint contains invalid optimizer values".into());
    }
    Ok((
        Mlp {
            input,
            hidden,
            parameters,
        },
        Optimizer {
            kind,
            learning_rate,
            beta1,
            beta2,
            eps,
            step_count,
            m: ms,
            v: vs,
        },
    ))
}

impl Optimizer {
    pub fn validate(&self, parameters: &[f64], gradient: &[f64]) -> Result<(), String> {
        if parameters.is_empty()
            || parameters.len() != gradient.len()
            || parameters.len() != self.m.len()
            || parameters.len() != self.v.len()
        {
            return Err("optimizer state length mismatch".into());
        }
        if !self.learning_rate.is_finite()
            || self.learning_rate <= 0.
            || !self.eps.is_finite()
            || self.eps <= 0.
            || ![self.beta1, self.beta2]
                .iter()
                .all(|v| v.is_finite() && (0.0..1.0).contains(v))
        {
            return Err("invalid optimizer settings".into());
        }
        if parameters
            .iter()
            .chain(gradient)
            .chain(&self.m)
            .chain(&self.v)
            .any(|v| !v.is_finite())
            || (self.kind == Kind::Adam && self.v.iter().any(|&v| v < 0.))
        {
            return Err("nonfinite values or negative second moments".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_validation_and_safe_install() -> Result<(), String> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("s02-checkpoint-{}-{stamp}", std::process::id()));
        fs::create_dir(&dir).map_err(|e| e.to_string())?;
        let path = dir.join("model.bin");
        let model = Mlp::new(3, 2)?;
        let optimizer = Optimizer::new(Kind::Adam, model.parameters.len());
        save(&path, &model, &optimizer)?;
        let original = fs::read(&path).map_err(|e| e.to_string())?;
        for field in [
            "version",
            "dimensions",
            "negative second moment",
            "trailing",
        ] {
            let mut bytes = original.clone();
            match field {
                "version" => bytes[8..12].copy_from_slice(&99u32.to_le_bytes()),
                "dimensions" => bytes[12..20].copy_from_slice(&u64::MAX.to_le_bytes()),
                "negative second moment" => {
                    let at = 93 + 16 * model.parameters.len();
                    bytes[at..at + 8].copy_from_slice(&(-1f64).to_le_bytes());
                }
                _ => bytes.push(0),
            }
            fs::write(&path, bytes).map_err(|e| e.to_string())?;
            assert!(restore(&path).is_err(), "accepted {field}");
        }
        fs::write(&path, &original).map_err(|e| e.to_string())?;
        let temporary = dir.join("model.bin.tmp");
        fs::write(&temporary, b"keep").map_err(|e| e.to_string())?;
        assert!(save(&path, &model, &optimizer).is_err());
        assert_eq!(fs::read(&path).map_err(|e| e.to_string())?, original);
        assert_eq!(fs::read(&temporary).map_err(|e| e.to_string())?, b"keep");
        fs::remove_dir_all(dir).map_err(|e| e.to_string())?;
        Ok(())
    }
}
