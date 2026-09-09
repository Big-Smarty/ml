//! Supplied model storage, stable scoring, and evaluation shared by the digit lab.
use crate::data::{Dataset, CLASSES};
#[derive(Clone, Debug)]
pub struct Linear {
    pub input: usize,
    pub weights: Vec<f64>,
    pub bias: [f64; CLASSES],
}
impl Linear {
    pub fn new(input: usize) -> Result<Self, String> {
        if input == 0 {
            return Err("input width must be positive".into());
        }
        Ok(Self {
            input,
            weights: vec![
                0.;
                input
                    .checked_mul(CLASSES)
                    .ok_or("parameter count overflow")?
            ],
            bias: [0.; CLASSES],
        })
    }
    pub fn logits(&self, x: &[f64]) -> Result<[f64; CLASSES], String> {
        if x.len() != self.input || x.iter().any(|v| !v.is_finite()) {
            return Err("image must be finite with model input width".into());
        }
        let mut z = self.bias;
        for (c, z) in z.iter_mut().enumerate() {
            for (i, &v) in x.iter().enumerate() {
                *z += self.weights[c * self.input + i] * v;
            }
        }
        if z.iter().any(|v| !v.is_finite()) {
            return Err("linear forward overflow".into());
        }
        Ok(z)
    }
}
pub fn objective(logits: [f64; CLASSES], target: usize) -> Result<(f64, [f64; CLASSES]), String> {
    if target >= CLASSES || logits.iter().any(|v| !v.is_finite()) {
        return Err("finite logits and target 0..9 required".into());
    }
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut p = logits.map(|v| (v - maximum).exp());
    let sum = p.iter().sum::<f64>();
    for v in &mut p {
        *v /= sum;
    }
    let loss = (maximum - logits[target]) + sum.ln();
    if !loss.is_finite() {
        return Err("cross-entropy overflow".into());
    }
    Ok((loss, p))
}
pub fn predict(z: [f64; CLASSES]) -> usize {
    // All ten positions exist; on ties choose the first class consistently.
    (1..CLASSES).fold(0, |best, c| if z[c] > z[best] { c } else { best })
}
pub type Confusion = [[usize; CLASSES]; CLASSES];
#[derive(Debug)]
pub struct Evaluation {
    pub loss: f64,
    pub accuracy: f64,
    pub confusion: Confusion,
}
pub fn evaluate(
    data: &Dataset,
    logits: impl Fn(&[f64]) -> Result<[f64; CLASSES], String>,
    verbose: bool,
) -> Result<Evaluation, String> {
    if data.len() == 0 {
        return Err("evaluation needs images".into());
    }
    let mut confusion = [[0usize; CLASSES]; CLASSES];
    let mut loss = 0.;
    let mut correct = 0;
    for n in 0..data.len() {
        let z = logits(data.image(n))?;
        let y = data.labels[n] as usize;
        let pred = predict(z);
        loss += objective(z, y)?.0;
        confusion[y][pred] += 1;
        correct += usize::from(y == pred);
        if verbose && y != pred {
            println!(
                "error row {n}: true={y}, predicted={pred}, confidence={:.3}",
                objective(z, y)?.1[pred]
            );
        }
    }
    if verbose {
        println!("confusion rows=true, columns=prediction:");
        for (c, row) in confusion.iter().enumerate() {
            println!("{c}: {row:?}");
        }
    }
    if !loss.is_finite() {
        return Err("evaluation loss overflow".into());
    }
    Ok(Evaluation {
        loss: loss / data.len() as f64,
        accuracy: correct as f64 / data.len() as f64,
        confusion,
    })
}
