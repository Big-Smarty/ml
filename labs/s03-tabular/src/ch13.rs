//! Learner: implement split policy and training-only median/scale/vocabulary fitting.
//! transform is supplied and demonstrates why fitted state must travel with a model.
use crate::data::{self, Point, RawRow, Result};
#[derive(Clone, Debug, PartialEq)]
pub struct Prep {
    pub median: [f64; 3],
    pub mean: [f64; 3],
    pub scale: [f64; 3],
    pub categories: Vec<String>,
}
#[derive(Clone, Copy, Debug)]
pub enum Split {
    Row,
    Group,
    Time,
    FutureGroup,
}
pub type Fitter = fn(&[RawRow]) -> Result<Prep>;
pub type Splitter = fn(&[RawRow], Split) -> Result<(Vec<RawRow>, Vec<RawRow>)>;
pub fn validate(rows: &[RawRow]) -> Result<()> {
    if rows.is_empty()
        || rows
            .iter()
            .any(|r| r.sensors.iter().flatten().any(|v| !v.is_finite()) || r.label > 1)
    {
        return Err("need nonempty rows with finite observed sensors and binary labels".into());
    }
    Ok(())
}
pub fn fit(rows: &[RawRow]) -> Result<Prep> {
    validate(rows)?;
    let mut center = [0.; 3];
    for (j, c) in center.iter_mut().enumerate() {
        let values: Vec<_> = rows.iter().filter_map(|r| r.sensors[j]).collect();
        if values.is_empty() {
            return Err(format!("sensor {j} has no observed training values"));
        }
        *c = data::finite(values.iter().sum::<f64>() / values.len() as f64)?;
    }
    // Working numeric baseline: mean imputation, original units, no learned vocabulary.
    Ok(Prep {
        median: center,
        mean: [0.; 3],
        scale: [1.; 3],
        categories: vec![],
    })
}
pub fn split(rows: &[RawRow], _kind: Split) -> Result<(Vec<RawRow>, Vec<RawRow>)> {
    validate(rows)?;
    // A repeatable row holdout works mechanically but answers the wrong deployment question.
    Ok(rows.iter().cloned().partition(|r| r.id % 4 != 0))
}
impl Prep {
    pub fn transform(&self, row: &RawRow) -> Result<Point> {
        let mut features = Vec::new();
        for (j, value) in row.sensors.iter().enumerate() {
            if !self.scale[j].is_finite() || self.scale[j] <= 0. {
                return Err("invalid fitted scale".into());
            }
            features.push(data::finite(
                (value.unwrap_or(self.median[j]) - self.mean[j]) / self.scale[j],
            )?);
        }
        features.extend(row.sensors.iter().map(|v| f64::from(v.is_none())));
        features.extend(self.categories.iter().map(|c| f64::from(c == &row.regime)));
        // Unknown is a column fixed at fitting, never a new test-derived category.
        features.push(f64::from(!self.categories.contains(&row.regime)));
        Ok(Point {
            features,
            label: row.label,
        })
    }
    pub fn transform_all(&self, rows: &[RawRow]) -> Result<Vec<Point>> {
        rows.iter().map(|r| self.transform(r)).collect()
    }
}
pub fn run(_: &[String]) -> Result<()> {
    report(fit, split)
}
pub fn check() -> Result<()> {
    verify(fit, split)
}
pub fn report(fit: Fitter, split: Splitter) -> Result<()> {
    let rows = data::development(&data::load()?);
    println!("Maintenance v1; prediction at day d, target/repair known at d+1. Candidate features: 3 sensors, missingness, regime.");
    for kind in [Split::Row, Split::Group, Split::Time, Split::FutureGroup] {
        let (train, valid) = split(&rows, kind)?;
        if train.is_empty() || valid.is_empty() {
            return Err("each demonstration split needs both sides".into());
        }
        let prep = fit(&train)?;
        let x = prep.transform_all(&train)?;
        let v = prep.transform_all(&valid)?;
        let p = v
            .iter()
            .map(|r| crate::solutions::ch14::knn(&x, &r.features, 3))
            .collect::<Result<Vec<_>>>()?;
        crate::evaluation::show(&format!("{kind:?} k=3"), &data::predictions(&valid, &p)?)?;
        let overlap = valid
            .iter()
            .filter(|r| train.iter().any(|t| t.machine == r.machine))
            .count();
        println!(
            " train={} valid={} validation rows with known machines={} median={:?} vocabulary={:?}",
            train.len(),
            valid.len(),
            overlap,
            prep.median,
            prep.categories
        );
    }
    let (train, valid) = crate::solutions::ch13::split(&rows, Split::FutureGroup)?;
    let leaky: Vec<_> = valid
        .iter()
        .map(|r| f64::from(r.repair_after == "replaced"))
        .collect();
    crate::evaluation::show(
        "FORBIDDEN repair_after feature",
        &data::predictions(&valid, &leaky)?,
    )?;
    println!("Train-only temperature center {:.3}; fit-before-split center {:.3}. Perfect repair prediction uses tomorrow's answer.",fit(&train)?.mean[0],fit(&rows)?.mean[0]);
    Ok(())
}
pub fn verify(fit: Fitter, split: Splitter) -> Result<()> {
    let mut rows = data::development(&data::load()?);
    // Fit goals are checked first so the earlier sessions have an observable checkpoint.
    let train: Vec<_> = rows
        .iter()
        .filter(|r| r.machine < 12 && r.day <= 3)
        .cloned()
        .collect();
    let seed = &train[0];
    let tiny: Vec<_> = [Some(2.), Some(4.), Some(100.), None]
        .into_iter()
        .enumerate()
        .map(|(id, x)| RawRow {
            id,
            sensors: [x, Some(2.), Some(1.)],
            regime: if id % 2 == 0 { "a".into() } else { "b".into() },
            ..seed.clone()
        })
        .collect();
    let p = fit(&tiny)?;
    if p.median[0] != 4.
        || p.categories != ["a", "b"]
        || (p.mean[0] - 27.5).abs() > 1e-12
        || p.scale[1] != 1.
    {
        return data::goal(format!("Goal not met: training median4, imputed mean27.5, vocabulary[a,b], constant scale1; got {p:?}"));
    }
    let unknown = RawRow {
        regime: "new".into(),
        ..tiny[3].clone()
    };
    let q = p.transform(&unknown)?.features;
    if q.len() != 9 || q[3] != 1. || q[8] != 1. || q[6] != 0. || q[7] != 0. {
        return data::goal("unknown and missing indicators must preserve a fixed shape".into());
    }
    println!("Numeric fitting, vocabulary, missingness and unknown-category checkpoint passed.");
    for kind in [Split::Row, Split::Group, Split::Time, Split::FutureGroup] {
        let (a, b) = split(&rows, kind)?;
        let expected_train: Vec<_> = rows
            .iter()
            .filter(|r| match kind {
                Split::Row => r.id % 4 != 0,
                Split::Group => r.machine % 3 != 2,
                Split::Time => r.day <= 3,
                Split::FutureGroup => r.machine < 12 && r.day <= 3,
            })
            .map(|r| r.id)
            .collect();
        let expected_valid: Vec<_> = rows
            .iter()
            .filter(|r| match kind {
                Split::Row => r.id % 4 == 0,
                Split::Group => r.machine % 3 == 2,
                Split::Time => (5..=7).contains(&r.day),
                Split::FutureGroup => (12..18).contains(&r.machine) && (5..=7).contains(&r.day),
            })
            .map(|r| r.id)
            .collect();
        let mut actual_train: Vec<_> = a.iter().map(|r| r.id).collect();
        let mut actual_valid: Vec<_> = b.iter().map(|r| r.id).collect();
        actual_train.sort_unstable();
        actual_valid.sort_unstable();
        if actual_train != expected_train || actual_valid != expected_valid {
            return data::goal(format!("{kind:?} membership differs: expected train{} / valid{}, got train{} / valid{}; inspect machine and time predicates", expected_train.len(), expected_valid.len(), a.len(), b.len()));
        }
    }
    let original = fit(&train)?;
    for r in &mut rows {
        if r.machine >= 12 {
            r.sensors[0] = Some(9999.);
            r.regime = "held-only".into();
        }
    }
    let (new_train, _) = split(&rows, Split::FutureGroup)?;
    if fit(&new_train)? != original {
        return data::goal("held-out rows changed fitted state".into());
    }
    println!("Split, training-only state, missingness, unknown-category and held-only perturbation checks passed.");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_and_solution() -> Result<()> {
        let d = data::development(&data::load()?);
        assert!(fit(&d)?
            .transform(&d[0])?
            .features
            .iter()
            .all(|v| v.is_finite()));
        assert!(fit(&[]).is_err());
        crate::solutions::ch13::check()
    }
}
