//! Freeze membership by machine ID, then tune only on validation probabilities.
use crate::ch05::{self, Record, Splits};
pub fn split(rows: &[Record]) -> Result<Splits, String> {
    ch05::validate_records(rows)?;
    let mut parts: [Vec<Record>; 3] = std::array::from_fn(|_| Vec::new());
    for row in rows {
        let part = match row.machine % 5 {
            0 => 1,
            1 => 2,
            _ => 0,
        };
        parts[part].push(*row);
    }
    if parts.iter().any(Vec::is_empty) {
        return Err("need machines in every partition".into());
    }
    Ok(parts)
}
pub fn choose_threshold(
    validation: &[(f64, bool)],
    candidates: &[f64],
    miss_cost: usize,
) -> Result<f64, String> {
    let mut best = None;
    for &threshold in candidates {
        let cost = ch05::evaluate(validation, threshold)?.cost(miss_cost);
        if best.is_none_or(|(_, best_cost)| cost < best_cost) {
            best = Some((threshold, cost));
        }
    }
    best.map(|(t, _)| t)
        .ok_or("provide threshold candidates".into())
}
pub fn run(args: &[String]) -> Result<(), String> {
    ch05::report(split, choose_threshold, args)
}
pub fn check() -> Result<(), String> {
    ch05::verify(split, choose_threshold)
}
