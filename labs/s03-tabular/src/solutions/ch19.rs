//! The validation population is later readings from held machines. Three rotations
//! give each eligible later row one validation prediction; early rows train only.
use crate::ch19::{self, Config, Fold, Trial};
use crate::data::{RawRow, Result};
pub fn folds(rows: &[RawRow]) -> Result<Vec<Fold>> {
    crate::ch13::validate(rows)?;
    let mut ordered = rows.to_vec();
    ordered.sort_by_key(|r| r.id);
    let ids: std::collections::BTreeSet<_> = ordered.iter().map(|r| r.id).collect();
    if ids.len() != ordered.len() {
        return Err("stable IDs must be unique".into());
    }
    let mut folds = Vec::new();
    for held in 0..3 {
        let train = ordered
            .iter()
            .filter(|r| r.machine < 18 && r.machine / 6 != held && r.day <= 3)
            .cloned()
            .collect::<Vec<_>>();
        let valid = ordered
            .iter()
            .filter(|r| r.machine < 18 && r.machine / 6 == held && (5..=7).contains(&r.day))
            .cloned()
            .collect::<Vec<_>>();
        if train.is_empty() || valid.is_empty() {
            return Err("each fold needs early training and later held-machine rows".into());
        }
        folds.push(Fold { train, valid });
    }
    Ok(folds)
}
pub fn search(folds: &[Fold], candidates: &[Config]) -> Result<(usize, Vec<Trial>)> {
    if candidates.is_empty() {
        return Err("need frozen candidates".into());
    }
    let mut trials: Vec<Trial> = Vec::new();
    let mut best = 0;
    for &candidate in candidates {
        let trial = ch19::trial(folds, candidate)?;
        // Strict improvement preserves the declared order when measured costs tie.
        if trials.is_empty() || trial.metrics.cost() < trials[best].metrics.cost() {
            best = trials.len();
        }
        trials.push(trial);
    }
    Ok((best, trials))
}
pub fn run(args: &[String]) -> Result<()> {
    ch19::report(folds, search, args.iter().any(|a| a == "--final"))
}
pub fn check() -> Result<()> {
    ch19::verify(folds, search)
}
