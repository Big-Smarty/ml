//! Supplied schema, parsing, deterministic draws, and numeric boundaries.
use std::collections::BTreeSet;
pub type Result<T> = std::result::Result<T, String>;
pub const CSV: &str = include_str!("../data/maintenance.csv");
#[derive(Clone, Debug, PartialEq)]
pub struct RawRow {
    pub id: usize,
    pub machine: usize,
    pub day: usize,
    pub sensors: [Option<f64>; 3],
    pub regime: String,
    pub label: u8,
    pub repair_after: String,
    pub probability: f64,
}
#[derive(Clone, Debug)]
pub struct Point {
    pub features: Vec<f64>,
    pub label: u8,
}
#[derive(Clone, Copy, Debug)]
pub struct Prediction {
    pub id: usize,
    pub group: usize,
    pub probability: f64,
    pub label: u8,
}
pub fn parse(text: &str) -> Result<Vec<RawRow>> {
    let mut lines = text.lines();
    if lines.next() != CSV.lines().next() {
        return Err("expected the maintenance v1 ten-column header".into());
    }
    let mut rows = Vec::new();
    let mut ids = BTreeSet::new();
    for (line, text) in lines.enumerate() {
        let fields: Vec<_> = text.split(',').collect();
        if fields.len() != 10 {
            return Err(format!("line {}: expected ten unquoted fields", line + 2));
        }
        let integer = |i: usize| {
            fields[i]
                .parse::<usize>()
                .map_err(|_| format!("line {}: invalid integer column {i}", line + 2))
        };
        let mut sensors = [None; 3];
        for (sensor, field) in sensors.iter_mut().zip(&fields[3..6]) {
            if !field.is_empty() {
                let value = field
                    .parse::<f64>()
                    .map_err(|_| format!("line {}: invalid sensor", line + 2))?;
                if !value.is_finite() {
                    return Err("sensor must be finite or blank".into());
                }
                *sensor = Some(value);
            }
        }
        let id = integer(0)?;
        let label = integer(7)?;
        let probability = fields[9]
            .parse::<f64>()
            .map_err(|_| "invalid probability")?;
        if !ids.insert(id)
            || label > 1
            || fields[6].is_empty()
            || !["none", "replaced"].contains(&fields[8])
            || !(0.0..=1.0).contains(&probability)
        {
            return Err("duplicate ID, invalid label, regime, repair state, or probability".into());
        }
        rows.push(RawRow {
            id,
            machine: integer(1)?,
            day: integer(2)?,
            sensors,
            regime: fields[6].into(),
            label: label as u8,
            repair_after: fields[8].into(),
            probability,
        });
    }
    if rows.is_empty() {
        return Err("dataset must contain rows".into());
    }
    Ok(rows)
}
pub fn load() -> Result<Vec<RawRow>> {
    parse(CSV)
}
pub fn development(rows: &[RawRow]) -> Vec<RawRow> {
    rows.iter()
        .filter(|r| r.machine < 18 && r.day <= 7)
        .cloned()
        .collect()
}
pub fn final_test(rows: &[RawRow]) -> Vec<RawRow> {
    rows.iter()
        .filter(|r| r.machine >= 18 && r.day >= 9)
        .cloned()
        .collect()
}
pub fn fingerprint() -> u64 {
    CSV.bytes().fold(0xcbf29ce484222325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}
pub fn validate_points(rows: &[Point]) -> Result<usize> {
    let d = rows
        .first()
        .ok_or("need nonempty training data")?
        .features
        .len();
    if d == 0
        || rows.iter().any(|r| {
            r.features.len() != d || r.label > 1 || r.features.iter().any(|v| !v.is_finite())
        })
    {
        return Err("features need a fixed nonzero width, finite values, and binary labels".into());
    }
    Ok(d)
}
pub fn validate_vectors(rows: &[Vec<f64>], minimum: usize) -> Result<usize> {
    let d = rows.first().ok_or("need nonempty vectors")?.len();
    if rows.len() < minimum
        || d == 0
        || rows
            .iter()
            .any(|r| r.len() != d || r.iter().any(|v| !v.is_finite()))
    {
        return Err("invalid vector count, shape, or value".into());
    }
    Ok(d)
}
pub fn query(x: &[f64], d: usize) -> Result<()> {
    if x.len() != d || x.iter().any(|v| !v.is_finite()) {
        Err("query shape or values are invalid".into())
    } else {
        Ok(())
    }
}
pub fn finite(value: f64) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err("arithmetic overflow; rescale the inputs".into())
    }
}
pub fn distance(a: &[f64], b: &[f64]) -> Result<f64> {
    query(a, b.len())?;
    query(b, a.len())?;
    finite(a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum())
}
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
#[derive(Clone, Copy)]
pub struct Rng(pub u64);
impl Rng {
    pub fn index(&mut self, n: usize) -> usize {
        // Callers establish n>0; this RNG is supplied simulation plumbing, not cryptography.
        self.0 = self.0.max(1);
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 as usize % n
    }
}
pub fn predictions(raw: &[RawRow], p: &[f64]) -> Result<Vec<Prediction>> {
    if raw.len() != p.len() {
        return Err("one probability per row required".into());
    }
    Ok(raw
        .iter()
        .zip(p)
        .map(|(r, &probability)| Prediction {
            id: r.id,
            group: r.machine,
            probability,
            label: r.label,
        })
        .collect())
}
/// A numerical learning goal failed; distinct from malformed input or runtime failure.
pub fn goal<T>(message: String) -> Result<T> {
    Err(format!("GOAL_NOT_MET: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixture_and_boundaries() -> Result<()> {
        let rows = load()?;
        assert_eq!(rows.len(), 288);
        assert_eq!(development(&rows).len(), 144);
        assert_eq!(final_test(&rows).len(), 18);
        assert!(rows.iter().any(|r| r.sensors.contains(&None)));
        assert!(parse("").is_err());
        assert!(parse(&(CSV.to_owned() + CSV.lines().nth(1).unwrap())).is_err());
        assert!(parse(&CSV.replacen("48", "NaN", 1)).is_err());
        assert!(distance(&[f64::MAX], &[-f64::MAX]).is_err());
        assert!(validate_points(&[Point {
            features: vec![0.],
            label: 2
        }])
        .is_err());
        Ok(())
    }
}
