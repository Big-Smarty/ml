//! Complete groups are the sampling unit. Duplicated within-machine rows carry
//! their pairing and dependence into every replicate; they are not new machines.
use crate::ch12::Bin;
use crate::data::{Prediction, Result, Rng};
use crate::evaluation::{metrics, validate};
use std::collections::BTreeMap;
pub fn interval(rows: &[Prediction], repetitions: usize, seed: u64) -> Result<(f64, f64)> {
    validate(rows)?;
    if !(40..=100_000).contains(&repetitions) {
        return Err("repetitions must be in 40..=100000".into());
    }
    let mut groups: BTreeMap<usize, Vec<Prediction>> = BTreeMap::new();
    for &row in rows {
        groups.entry(row.group).or_default().push(row);
    }
    let groups: Vec<_> = groups.values().collect();
    let mut rng = Rng(seed);
    let mut estimates = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let mut sample = Vec::new();
        for _ in 0..groups.len() {
            sample.extend(groups[rng.index(groups.len())]);
        }
        estimates.push(metrics(&sample)?.accuracy());
    }
    estimates.sort_by(f64::total_cmp);
    // Empirical percentile convention: floor(q*B), zero-based. Coverage is approximate.
    Ok((
        estimates[repetitions * 25 / 1000],
        estimates[repetitions * 975 / 1000],
    ))
}
pub fn calibration(rows: &[Prediction], bins: usize) -> Result<Vec<Bin>> {
    validate(rows)?;
    if bins == 0 || bins > 100 {
        return Err("bins must be in 1..=100".into());
    }
    let mut totals = vec![(0, 0., 0.); bins];
    for row in rows {
        let i = ((row.probability * bins as f64) as usize).min(bins - 1);
        totals[i].0 += 1;
        totals[i].1 += row.probability;
        totals[i].2 += f64::from(row.label);
    }
    Ok(totals
        .into_iter()
        .enumerate()
        .filter(|(_, b)| b.0 > 0)
        .map(|(index, (count, p, y))| Bin {
            index,
            count,
            probability: p / count as f64,
            frequency: y / count as f64,
        })
        .collect())
}
pub fn run(_: &[String]) -> Result<()> {
    crate::ch12::report(interval, calibration)
}
pub fn check() -> Result<()> {
    crate::ch12::verify(interval, calibration)
}
