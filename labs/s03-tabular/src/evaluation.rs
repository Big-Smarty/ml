//! Supplied reporting: all model and split choices remain in the chapter files.
use crate::data::{Prediction, Result};
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Metrics {
    pub correct: usize,
    pub total: usize,
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub tn: usize,
    pub brier: f64,
}
pub fn validate(rows: &[Prediction]) -> Result<()> {
    if rows.is_empty()
        || rows
            .iter()
            .any(|r| !(0.0..=1.0).contains(&r.probability) || r.label > 1)
    {
        Err("need paired finite probabilities in [0,1] and binary outcomes".into())
    } else {
        Ok(())
    }
}
pub fn metrics(rows: &[Prediction]) -> Result<Metrics> {
    validate(rows)?;
    let mut m = Metrics {
        correct: 0,
        total: rows.len(),
        tp: 0,
        fp: 0,
        fn_: 0,
        tn: 0,
        brier: 0.,
    };
    for r in rows {
        match (r.probability >= 0.5, r.label == 1) {
            (true, true) => m.tp += 1,
            (true, false) => m.fp += 1,
            (false, true) => m.fn_ += 1,
            (false, false) => m.tn += 1,
        }
        m.brier += (r.probability - f64::from(r.label)).powi(2);
    }
    m.correct = m.tp + m.tn;
    m.brier /= m.total as f64;
    Ok(m)
}
impl Metrics {
    pub fn accuracy(self) -> f64 {
        self.correct as f64 / self.total as f64
    }
    pub fn cost(self) -> f64 {
        (4 * self.fn_ + self.fp) as f64 / self.total as f64
    }
}
pub fn show(name: &str, rows: &[Prediction]) -> Result<()> {
    let m = metrics(rows)?;
    println!(
        "{name}: n={} accuracy={:.3} Brier={:.4} cost(4FN+FP)/n={:.3} TP={} FP={} FN={} TN={}",
        m.total,
        m.accuracy(),
        m.brier,
        m.cost(),
        m.tp,
        m.fp,
        m.fn_,
        m.tn
    );
    Ok(())
}
