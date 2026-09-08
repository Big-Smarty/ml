//! A working SGD classifier, ready to gain momentum.
use std::{env, fs};
const CLASSES: usize = 10;
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

#[cfg(test)]
fn momentum_step(parameter: &mut f64, velocity: &mut f64, gradient: f64, rate: f64, beta: f64) {
    let _ = (parameter, velocity, gradient, rate, beta);
    // TODO: update velocity from the old velocity and gradient, then the parameter.
    todo!("implement momentum")
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    let data = match args.as_slice() {
        [_] => fixture(true),
        [_, images, labels] => load(images, labels)?,
        _ => return Err("usage: ch11-starter [IMAGES_IDX LABELS_IDX]".into()),
    };
    let mut model = Mlp::new(data.width(), 16);
    let before = model.metrics(&data)?.0;
    // A small batch and familiar SGD keep this checkpoint quick even with MNIST.
    for _ in 0..20 {
        let (_, gradients) = model.loss_grad(&data, 0, data.len().min(32))?;
        for (weight, gradient) in model.p.iter_mut().zip(gradients) {
            *weight -= 0.1 * gradient;
        }
    }
    println!(
        "SGD checkpoint: training loss {before:.4} -> {:.4}",
        model.metrics(&data)?.0
    );
    println!("Complete momentum_step, allocate one velocity per parameter, and replace the SGD update above.");
    Ok(())
}
#[test]
fn first_momentum_step() {
    let (mut p, mut v) = (1.0, 0.0);
    momentum_step(&mut p, &mut v, 2.0, 0.1, 0.9);
    assert!((v - 2.0).abs() < 1e-12);
    assert!((p - 0.8).abs() < 1e-12);
    momentum_step(&mut p, &mut v, 0.0, 0.1, 0.9);
    assert!((v - 1.8).abs() < 1e-12);
    assert!((p - 0.62).abs() < 1e-12);
}
