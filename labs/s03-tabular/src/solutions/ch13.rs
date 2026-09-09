//! Split raw records first. Fit each statistic from that training slice only.
use crate::ch13::{validate, Prep, Split};
use crate::data::{self, RawRow, Result};
pub fn split(rows: &[RawRow], kind: Split) -> Result<(Vec<RawRow>, Vec<RawRow>)> {
    validate(rows)?;
    let mut train = Vec::new();
    let mut valid = Vec::new();
    for r in rows {
        let (a, b) = match kind {
            Split::Row => (r.id % 4 != 0, r.id % 4 == 0),
            Split::Group => (r.machine % 3 != 2, r.machine % 3 == 2),
            Split::Time => (r.day <= 3, (5..=7).contains(&r.day)),
            Split::FutureGroup => (
                r.machine < 12 && r.day <= 3,
                (12..18).contains(&r.machine) && (5..=7).contains(&r.day),
            ),
        };
        if a {
            train.push(r.clone());
        }
        if b {
            valid.push(r.clone());
        }
    }
    if train.is_empty() || valid.is_empty() {
        return Err("split produced an empty partition".into());
    }
    Ok((train, valid))
}
pub fn fit(rows: &[RawRow]) -> Result<Prep> {
    validate(rows)?;
    let mut median = [0.; 3];
    let mut mean = [0.; 3];
    let mut scale = [0.; 3];
    for j in 0..3 {
        let mut values: Vec<_> = rows.iter().filter_map(|r| r.sensors[j]).collect();
        if values.is_empty() {
            return Err(format!("sensor{j} has no observed training value"));
        }
        values.sort_by(f64::total_cmp);
        let middle = values.len() / 2;
        median[j] = if values.len() % 2 == 1 {
            values[middle]
        } else {
            values[middle - 1].midpoint(values[middle])
        };
        // Scaling follows imputation, so fit and transform see the same numeric values.
        mean[j] = data::finite(
            rows.iter()
                .map(|r| r.sensors[j].unwrap_or(median[j]) / rows.len() as f64)
                .sum(),
        )?;
        scale[j] = data::finite(
            (rows
                .iter()
                .map(|r| (r.sensors[j].unwrap_or(median[j]) - mean[j]).powi(2) / rows.len() as f64)
                .sum::<f64>())
            .sqrt(),
        )?;
        if scale[j] == 0. {
            scale[j] = 1.;
        }
    }
    let mut categories: Vec<_> = rows.iter().map(|r| r.regime.clone()).collect();
    categories.sort();
    categories.dedup();
    Ok(Prep {
        median,
        mean,
        scale,
        categories,
    })
}
pub fn run(_: &[String]) -> Result<()> {
    crate::ch13::report(fit, split)
}
pub fn check() -> Result<()> {
    crate::ch13::verify(fit, split)
}
