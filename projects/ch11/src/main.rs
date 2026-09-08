//! A real one-hidden-layer classifier with momentum, Adam, and resumable checkpoints.
use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
};
const CLASSES: usize = 10;
const MAGIC: &[u8; 8] = b"MLCH11\0\0";

#[derive(Clone, Debug)]
struct Dataset {
    x: Vec<f64>,
    y: Vec<u8>,
    rows: usize,
    cols: usize,
}
impl Dataset {
    fn len(&self) -> usize {
        self.y.len()
    }
    fn width(&self) -> usize {
        self.rows * self.cols
    }
    fn row(&self, n: usize) -> &[f64] {
        let d = self.width();
        &self.x[n * d..(n + 1) * d]
    }
}
fn u32be(b: &[u8], p: usize) -> Result<usize, String> {
    Ok(u32::from_be_bytes(
        b.get(p..p + 4)
            .ok_or("truncated IDX header")?
            .try_into()
            .unwrap(),
    ) as usize)
}
fn parse_idx(i: &[u8], l: &[u8]) -> Result<Dataset, String> {
    if u32be(i, 0)? != 2051 || u32be(l, 0)? != 2049 {
        return Err("expected IDX image magic 2051 and label magic 2049".into());
    }
    let (n, nl, r, c) = (u32be(i, 4)?, u32be(l, 4)?, u32be(i, 8)?, u32be(i, 12)?);
    if n == 0 || r == 0 || c == 0 || n != nl {
        return Err("IDX counts/dimensions are invalid".into());
    }
    let z = n
        .checked_mul(r)
        .and_then(|v| v.checked_mul(c))
        .ok_or("IDX size overflow")?;
    let image_len = z.checked_add(16).ok_or("IDX image length overflow")?;
    let label_len = n.checked_add(8).ok_or("IDX label length overflow")?;
    if i.len() != image_len || l.len() != label_len {
        return Err("IDX payload length does not match header".into());
    }
    if l[8..].iter().any(|&v| v as usize >= CLASSES) {
        return Err("IDX label is outside 0..9".into());
    }
    Ok(Dataset {
        x: i[16..].iter().map(|&v| v as f64 / 255.).collect(),
        y: l[8..].to_vec(),
        rows: r,
        cols: c,
    })
}
fn load(i: &str, l: &str) -> Result<Dataset, String> {
    let ib = fs::read(i).map_err(|e| format!("cannot read {i}: {e}"))?;
    let lb = fs::read(l).map_err(|e| format!("cannot read {l}: {e}"))?;
    parse_idx(&ib, &lb)
}
fn fixture(train: bool) -> Dataset {
    let s = if train {
        vec![
            ([255, 255, 255, 255], 0),
            ([230, 255, 255, 230], 0),
            ([0, 255, 0, 255], 1),
            ([0, 230, 0, 255], 1),
            ([255, 255, 0, 0], 2),
            ([255, 230, 0, 0], 2),
        ]
    } else {
        vec![
            ([240, 240, 255, 255], 0),
            ([0, 255, 0, 240], 1),
            ([240, 255, 0, 0], 2),
        ]
    };
    let mut i = Vec::new();
    for v in [2051u32, s.len() as u32, 2, 2] {
        i.extend(v.to_be_bytes())
    }
    let mut l = Vec::new();
    for v in [2049u32, s.len() as u32] {
        l.extend(v.to_be_bytes())
    }
    for (a, y) in s {
        i.extend(a);
        l.push(y)
    }
    parse_idx(&i, &l).unwrap()
}

#[derive(Clone, Debug)]
struct Mlp {
    input: usize,
    hidden: usize,
    p: Vec<f64>,
}
impl Mlp {
    fn checked_count(input: usize, hidden: usize) -> Option<usize> {
        hidden
            .checked_mul(input)?
            .checked_add(hidden)?
            .checked_add(CLASSES.checked_mul(hidden)?)?
            .checked_add(CLASSES)
    }
    fn count(input: usize, hidden: usize) -> usize {
        Self::checked_count(input, hidden).expect("model dimensions are too large")
    }
    fn new(input: usize, hidden: usize) -> Self {
        let mut seed = 7u64;
        let mut p = vec![0.; Self::count(input, hidden)];
        let (b1, w2, _) = {
            let b1 = hidden * input;
            let w2 = b1 + hidden;
            (b1, w2, w2 + CLASSES * hidden)
        };
        let mut draw = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((seed >> 11) as f64 / (1u64 << 53) as f64) * 2. - 1.) * scale
        };
        for v in &mut p[..b1] {
            *v = draw((6. / input as f64).sqrt())
        }
        for v in &mut p[w2..w2 + CLASSES * hidden] {
            *v = draw((6. / hidden as f64).sqrt())
        }
        Self { input, hidden, p }
    }
    fn parts(&self) -> (usize, usize, usize) {
        let b1 = self.hidden * self.input;
        let w2 = b1 + self.hidden;
        let b2 = w2 + CLASSES * self.hidden;
        (b1, w2, b2)
    }
    fn forward(&self, x: &[f64]) -> (Vec<f64>, [f64; CLASSES]) {
        let (b1, w2, b2) = self.parts();
        let mut h = vec![0.; self.hidden];
        for (j, hidden) in h.iter_mut().enumerate() {
            let mut z = self.p[b1 + j];
            for (k, &input) in x.iter().enumerate() {
                z += self.p[j * self.input + k] * input
            }
            *hidden = z.max(0.)
        }
        let mut out = [0.; CLASSES];
        for (c, output) in out.iter_mut().enumerate() {
            *output = self.p[b2 + c];
            for (j, &hidden) in h.iter().enumerate() {
                *output += self.p[w2 + c * self.hidden + j] * hidden
            }
        }
        (h, out)
    }
    fn probs(z: [f64; CLASSES]) -> [f64; CLASSES] {
        let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut p = z.map(|v| (v - m).exp());
        let s = p.iter().sum::<f64>();
        for v in &mut p {
            *v /= s
        }
        p
    }
    fn cross_entropy(z: [f64; CLASSES], target: usize) -> f64 {
        let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (m - z[target]) + z.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
    }
    fn loss_grad(&self, d: &Dataset, start: usize, end: usize) -> Result<(f64, Vec<f64>), String> {
        if d.width() != self.input || start >= end || end > d.len() {
            return Err("invalid model/data batch".into());
        }
        let mut g = vec![0.; self.p.len()];
        let mut loss = 0.;
        let (b1, w2, b2) = self.parts();
        for n in start..end {
            let x = d.row(n);
            let (h, z) = self.forward(x);
            if h.iter().chain(&z).any(|v| !v.is_finite()) {
                return Err("forward computation overflowed".into());
            }
            loss += Self::cross_entropy(z, d.y[n] as usize);
            let mut dz = Self::probs(z);
            dz[d.y[n] as usize] -= 1.;
            let mut dh = vec![0.; self.hidden];
            for c in 0..CLASSES {
                g[b2 + c] += dz[c];
                for j in 0..self.hidden {
                    g[w2 + c * self.hidden + j] += dz[c] * h[j];
                    dh[j] += dz[c] * self.p[w2 + c * self.hidden + j]
                }
            }
            for j in 0..self.hidden {
                if h[j] > 0. {
                    g[b1 + j] += dh[j];
                    for k in 0..self.input {
                        g[j * self.input + k] += dh[j] * x[k]
                    }
                }
            }
        }
        let q = (end - start) as f64;
        for v in &mut g {
            *v /= q
        }
        if !loss.is_finite() || g.iter().any(|v| !v.is_finite()) {
            return Err("loss or gradient overflowed".into());
        }
        Ok((loss / q, g))
    }
    fn metrics(&self, d: &Dataset) -> Result<(f64, f64), String> {
        if d.width() != self.input || d.len() == 0 {
            return Err("evaluation shape mismatch".into());
        }
        let (mut loss, mut good) = (0., 0usize);
        for n in 0..d.len() {
            let (_, z) = self.forward(d.row(n));
            loss += Self::cross_entropy(z, d.y[n] as usize);
            let k = (0..CLASSES).max_by(|&a, &b| z[a].total_cmp(&z[b])).unwrap();
            good += (k == d.y[n] as usize) as usize
        }
        if !loss.is_finite() {
            return Err("evaluation loss overflowed".into());
        }
        Ok((loss / d.len() as f64, good as f64 / d.len() as f64))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Momentum,
    Adam,
}
#[derive(Clone, Debug)]
struct Optimizer {
    kind: Kind,
    rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    step: u64,
    m: Vec<f64>,
    v: Vec<f64>,
}
impl Optimizer {
    fn new(kind: Kind, n: usize) -> Self {
        let rate = if kind == Kind::Adam { 0.01 } else { 0.1 };
        Self {
            kind,
            rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            step: 0,
            m: vec![0.; n],
            v: vec![0.; n],
        }
    }
    fn update(&mut self, p: &mut [f64], g: &[f64]) -> Result<(), String> {
        if p.len() != g.len() || p.len() != self.m.len() || p.len() != self.v.len() {
            return Err("optimizer state length mismatch".into());
        }
        self.step = self.step.checked_add(1).ok_or("optimizer step overflow")?;
        for i in 0..p.len() {
            self.m[i] = self.beta1 * self.m[i]
                + (if self.kind == Kind::Adam {
                    1. - self.beta1
                } else {
                    1.
                }) * g[i];
            if self.kind == Kind::Adam {
                self.v[i] = self.beta2 * self.v[i] + (1. - self.beta2) * g[i] * g[i];
                let mh = self.m[i] / (1. - self.beta1.powf(self.step as f64));
                let vh = self.v[i] / (1. - self.beta2.powf(self.step as f64));
                p[i] -= self.rate * mh / (vh.sqrt() + self.eps)
            } else {
                p[i] -= self.rate * self.m[i]
            }
        }
        if p.iter()
            .chain(&self.m)
            .chain(&self.v)
            .any(|v| !v.is_finite())
        {
            return Err("optimizer produced nonfinite parameters or moment buffers".into());
        }
        Ok(())
    }
}
fn train(
    m: &mut Mlp,
    o: &mut Optimizer,
    d: &Dataset,
    epochs: usize,
    batch: usize,
) -> Result<(), String> {
    if epochs == 0 || batch == 0 {
        return Err("epochs and batch must be positive".into());
    }
    for _ in 0..epochs {
        for s in (0..d.len()).step_by(batch) {
            let (_, g) = m.loss_grad(d, s, (s + batch).min(d.len()))?;
            o.update(&mut m.p, &g)?
        }
    }
    Ok(())
}

fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend(v.to_le_bytes())
}
fn put_f64(b: &mut Vec<u8>, v: f64) {
    b.extend(v.to_le_bytes())
}
fn save(path: &Path, m: &Mlp, o: &Optimizer) -> Result<(), String> {
    let mut b = MAGIC.to_vec();
    b.extend(1u32.to_le_bytes());
    put_u64(&mut b, m.input as u64);
    put_u64(&mut b, m.hidden as u64);
    put_u64(&mut b, m.p.len() as u64);
    b.push(if o.kind == Kind::Adam { 1 } else { 0 });
    put_u64(&mut b, o.step);
    for v in [o.rate, o.beta1, o.beta2, o.eps] {
        put_f64(&mut b, v)
    }
    put_u64(&mut b, o.m.len() as u64);
    put_u64(&mut b, o.v.len() as u64);
    for a in [&m.p, &o.m, &o.v] {
        for &v in a {
            put_f64(&mut b, v)
        }
    }
    let tmp = path.with_extension("tmp");
    let mut f =
        fs::File::create(&tmp).map_err(|e| format!("cannot create checkpoint temp file: {e}"))?;
    f.write_all(&b)
        .and_then(|_| f.sync_all())
        .map_err(|e| format!("cannot write checkpoint: {e}"))?;
    fs::rename(&tmp, path).map_err(|e| format!("cannot install checkpoint: {e}"))
}
fn restore(path: &Path) -> Result<(Mlp, Optimizer), String> {
    let b = fs::read(path).map_err(|e| format!("cannot read checkpoint: {e}"))?;
    let mut p = 0usize;
    fn take<const N: usize>(b: &[u8], p: &mut usize) -> Result<[u8; N], String> {
        let end = p.checked_add(N).ok_or("checkpoint offset overflow")?;
        let s = b.get(*p..end).ok_or("truncated checkpoint")?;
        *p = end;
        Ok(s.try_into().unwrap())
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
    let step = u64::from_le_bytes(take(&b, &mut p)?);
    let rate = f64::from_le_bytes(take(&b, &mut p)?);
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
    if !rate.is_finite()
        || rate <= 0.
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
    let params = read_vec(n)?;
    let ms = read_vec(n)?;
    let vs = read_vec(n)?;
    if params.iter().chain(&ms).chain(&vs).any(|v| !v.is_finite())
        || (kind == Kind::Adam && vs.iter().any(|&v| v < 0.))
    {
        return Err("checkpoint contains invalid optimizer values".into());
    }
    Ok((
        Mlp {
            input,
            hidden,
            p: params,
        },
        Optimizer {
            kind,
            rate,
            beta1,
            beta2,
            eps,
            step,
            m: ms,
            v: vs,
        },
    ))
}
fn kind(s: &str) -> Result<Kind, String> {
    match s {
        "adam" => Ok(Kind::Adam),
        "momentum" => Ok(Kind::Momentum),
        _ => Err("optimizer must be adam or momentum".into()),
    }
}
fn run(
    tr: Dataset,
    held_out: Dataset,
    epochs: usize,
    k: Kind,
    checkpoint: Option<PathBuf>,
) -> Result<(), String> {
    if tr.rows != held_out.rows || tr.cols != held_out.cols {
        return Err("train and held-out dimensions differ".into());
    }
    if epochs == 0 && !checkpoint.as_ref().is_some_and(|p| p.is_file()) {
        return Err("zero-epoch evaluation requires an existing checkpoint".into());
    }
    let (mut m, mut o) = if let Some(ref p) = checkpoint {
        if p.exists() {
            let pair = restore(p)?;
            if pair.0.input != tr.width() {
                return Err("checkpoint input width differs from dataset".into());
            }
            pair
        } else {
            let m = Mlp::new(tr.width(), 16);
            let o = Optimizer::new(k, m.p.len());
            (m, o)
        }
    } else {
        let m = Mlp::new(tr.width(), 16);
        let o = Optimizer::new(k, m.p.len());
        (m, o)
    };
    let before = m.metrics(&held_out)?;
    if epochs == 0 {
        println!(
            "evaluation only: optimizer={:?}, step={}, held-out loss {:.4}, accuracy {:.1}%",
            o.kind,
            o.step,
            before.0,
            100. * before.1
        );
        return Ok(());
    }
    train(&mut m, &mut o, &tr, epochs, 32)?;
    let after = m.metrics(&held_out)?;
    if let Some(p) = checkpoint {
        save(&p, &m, &o)?;
        println!("checkpoint: {} (step {})", p.display(), o.step)
    }
    println!(
        "optimizer={:?}, held-out loss {:.4} -> {:.4}, accuracy {:.1}% -> {:.1}%",
        o.kind,
        before.0,
        after.0,
        100. * before.1,
        100. * after.1
    );
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = env::args().collect();
    match a.as_slice(){[_]=>{let p=env::temp_dir().join(format!("ch11-fixture-{}.ckpt",std::process::id()));run(fixture(true),fixture(false),300,Kind::Adam,Some(p.clone()))?;let(_,o)=restore(&p)?;println!("restored fixture checkpoint at step {}",o.step);fs::remove_file(p)?},[_,f,opt]if f=="--fixture"=>run(fixture(true),fixture(false),300,kind(opt)?,None)?,[_,f,ti,tl,vi,vl,e,opt,cp]if f=="--mnist"=>run(load(ti,tl)?,load(vi,vl)?,e.parse().map_err(|_|"epochs must be an integer")?,kind(opt)?,Some(cp.into()))?,_=>return Err("usage: ch11 [--fixture adam|momentum] | --mnist TRAIN_IMAGES TRAIN_LABELS TEST_IMAGES TEST_LABELS EPOCHS adam|momentum CHECKPOINT".into())}
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_optimizers_train_real_mlp() {
        for k in [Kind::Momentum, Kind::Adam] {
            let tr = fixture(true);
            let te = fixture(false);
            let mut m = Mlp::new(4, 8);
            let mut o = Optimizer::new(k, m.p.len());
            let a = m.metrics(&te).unwrap().0;
            train(&mut m, &mut o, &tr, 400, 3).unwrap();
            let (b, acc) = m.metrics(&te).unwrap();
            assert!(b < a && acc > 0.66)
        }
    }
    #[test]
    fn hidden_weight_gradient_matches_difference() {
        let d = fixture(true);
        let m = Mlp::new(4, 8);
        let (_, g) = m.loss_grad(&d, 0, 3).unwrap();
        let h = 1e-5;
        let mut p = m.clone();
        let mut n = m.clone();
        p.p[0] += h;
        n.p[0] -= h;
        let lp = p.loss_grad(&d, 0, 3).unwrap().0;
        let ln = n.loss_grad(&d, 0, 3).unwrap().0;
        let numeric = (lp - ln) / (2. * h);
        assert!((g[0] - numeric).abs() < 1e-6 + 1e-4 * numeric.abs())
    }
    #[test]
    fn checkpoint_restores_exact_optimizer_state() {
        let d = fixture(true);
        let mut a = Mlp::new(4, 8);
        let mut oa = Optimizer::new(Kind::Adam, a.p.len());
        train(&mut a, &mut oa, &d, 2, 3).unwrap();
        let p = env::temp_dir().join(format!("ch11-test-{}.ckpt", std::process::id()));
        save(&p, &a, &oa).unwrap();
        let (mut b, mut ob) = restore(&p).unwrap();
        let (_, ga) = a.loss_grad(&d, 0, 3).unwrap();
        let (_, gb) = b.loss_grad(&d, 0, 3).unwrap();
        oa.update(&mut a.p, &ga).unwrap();
        ob.update(&mut b.p, &gb).unwrap();
        assert_eq!(a.p, b.p);
        assert_eq!(oa.step, ob.step);
        fs::remove_file(p).unwrap()
    }
    #[test]
    fn corrupt_checkpoint_fails() {
        let p = env::temp_dir().join(format!("ch11-bad-{}.ckpt", std::process::id()));
        fs::write(&p, b"bad").unwrap();
        assert!(restore(&p).is_err());
        fs::remove_file(p).unwrap();
    }
    #[test]
    fn unsafe_checkpoint_fields_fail_before_state_allocation() {
        let model = Mlp::new(4, 8);
        let optimizer = Optimizer::new(Kind::Adam, model.p.len());
        let p = env::temp_dir().join(format!("ch11-unsafe-{}.ckpt", std::process::id()));
        save(&p, &model, &optimizer).unwrap();
        let mut bytes = fs::read(&p).unwrap();
        bytes[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
        fs::write(&p, &bytes).unwrap();
        assert!(restore(&p).is_err());
        fs::remove_file(p).unwrap();
    }
    #[test]
    fn shared_offset_and_evaluation_only_contract() {
        assert!((Mlp::cross_entropy([1e16; CLASSES], 0) - (CLASSES as f64).ln()).abs() < 1e-12);
        let model = Mlp::new(4, 8);
        let optimizer = Optimizer::new(Kind::Adam, model.p.len());
        let path = env::temp_dir().join(format!("ch11-eval-{}.ckpt", std::process::id()));
        save(&path, &model, &optimizer).unwrap();
        let original = fs::read(&path).unwrap();
        run(
            fixture(true),
            fixture(false),
            0,
            Kind::Momentum,
            Some(path.clone()),
        )
        .unwrap();
        assert_eq!(fs::read(&path).unwrap(), original);
        fs::remove_file(&path).unwrap();
        assert!(run(fixture(true), fixture(false), 0, Kind::Adam, Some(path)).is_err());
        assert!(run(fixture(true), fixture(false), 0, Kind::Adam, None).is_err());
        let mut reshaped = fixture(false);
        reshaped.rows = 1;
        reshaped.cols = 4;
        assert!(run(fixture(true), reshaped, 1, Kind::Adam, None).is_err());
    }
    #[test]
    fn finite_parameters_do_not_hide_overflowing_optimizer_state() {
        let mut p = [1.0];
        let mut optimizer = Optimizer::new(Kind::Adam, 1);
        assert!(optimizer.update(&mut p, &[1e200]).is_err());
        let mut model = Mlp::new(4, 8);
        model.p.fill(f64::MAX);
        assert!(model.metrics(&fixture(true)).is_err());
        assert!(model.loss_grad(&fixture(true), 0, 1).is_err());
    }
}
