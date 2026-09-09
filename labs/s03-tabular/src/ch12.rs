//! Learner: replace the row-normal interval with a whole-machine bootstrap;
//! replace the global calibration summary with bins. RNG and report are supplied.
use crate::data::{self, Prediction, Result};
use crate::evaluation::{metrics, show, validate};
#[derive(Debug, PartialEq)]
pub struct Bin {
    pub index: usize,
    pub count: usize,
    pub probability: f64,
    pub frequency: f64,
}
pub type Interval = fn(&[Prediction], usize, u64) -> Result<(f64, f64)>;
pub type Calibration = fn(&[Prediction], usize) -> Result<Vec<Bin>>;

pub fn interval(rows: &[Prediction], repetitions: usize, _seed: u64) -> Result<(f64, f64)> {
    validate(rows)?;
    if repetitions < 40 {
        return Err("use at least 40 repetitions".into());
    }
    // Working descriptive approximation: mistakenly treats each reading as independent.
    let p = metrics(rows)?.accuracy();
    let width = 1.96 * (p * (1. - p) / rows.len() as f64).sqrt();
    Ok(((p - width).max(0.), (p + width).min(1.)))
}
pub fn calibration(rows: &[Prediction], bins: usize) -> Result<Vec<Bin>> {
    validate(rows)?;
    if bins == 0 || bins > 100 {
        return Err("bins must be in 1..=100".into());
    }
    // Useful overall comparison; learner replaces it with separate nonempty bins.
    Ok(vec![Bin {
        index: 0,
        count: rows.len(),
        probability: rows.iter().map(|r| r.probability).sum::<f64>() / rows.len() as f64,
        frequency: rows.iter().map(|r| f64::from(r.label)).sum::<f64>() / rows.len() as f64,
    }])
}
pub fn run(_: &[String]) -> Result<()> {
    report(interval, calibration)
}
pub fn check() -> Result<()> {
    verify(interval, calibration)
}
pub fn report(interval: Interval, calibration: Calibration) -> Result<()> {
    let rows: Vec<_> = data::load()?
        .into_iter()
        .filter(|r| (12..18).contains(&r.machine) && (5..8).contains(&r.day))
        .map(|r| Prediction {
            id: r.id,
            group: r.machine,
            probability: r.probability,
            label: r.label,
        })
        .collect();
    println!("Maintenance v1: fixed sensor-rule predictions on 18 development readings from 6 machines; no model fitting.");
    show("frozen heuristic", &rows)?;
    println!(
        "95% interval {:?}; seed=7; 2000 repetitions requested",
        interval(&rows, 2000, 7)?
    );
    for bin in calibration(&rows, 4)? {
        println!(
            "bin {}: n={} mean p={:.4}, frequency={:.4}",
            bin.index, bin.count, bin.probability, bin.frequency
        );
    }
    println!("A working report is not a learning-goal pass. --check inspects groups, pairs, endpoints and bin arithmetic.");
    Ok(())
}
pub fn verify(interval: Interval, calibration: Calibration) -> Result<()> {
    let rows = [
        Prediction {
            id: 0,
            group: 0,
            probability: 0.1,
            label: 0,
        },
        Prediction {
            id: 1,
            group: 0,
            probability: 0.9,
            label: 1,
        },
        Prediction {
            id: 2,
            group: 1,
            probability: 0.8,
            label: 0,
        },
        Prediction {
            id: 3,
            group: 1,
            probability: 0.2,
            label: 1,
        },
    ];
    let endpoints = interval(&rows, 2000, 7)?;
    println!("two machines (one always correct, one always wrong): interval={endpoints:?}; expected [0,1]");
    if endpoints != (0., 1.) {
        return data::goal("Goal not met: sample whole machine groups with replacement and recompute accuracy; row-count standard errors invent independence".into());
    }
    let duplicated: Vec<_> = rows.iter().flat_map(|r| [*r, *r, *r]).collect();
    if interval(&duplicated, 2000, 7)? != endpoints {
        return data::goal(
            "Duplicating readings within each machine must not change this cluster interval".into(),
        );
    }
    let b = calibration(&rows, 2)?;
    if b.len() != 2
        || b[0].count != 2
        || (b[0].probability - 0.15).abs() > 1e-12
        || b[0].frequency != 0.5
        || (b[1].probability - 0.85).abs() > 1e-12
    {
        return data::goal("Goal not met: form separate nonempty reliability bins; expected (2,.15,.5) and (2,.85,.5)".into());
    }
    let certain = [Prediction {
        id: 7,
        group: 4,
        probability: 1.,
        label: 1,
    }];
    if interval(&certain, 100, 3)? != (1., 1.) || calibration(&certain, 4)?[0].index != 3 {
        return data::goal(
            "p=1 belongs in the last bin; identical correct rows give interval [1,1]".into(),
        );
    }
    if interval(&[], 100, 7).is_ok() || calibration(&rows, 0).is_ok() {
        return data::goal("reject invalid statistical inputs".into());
    }
    println!(
        "Goal checks passed. Explain why six machines still cannot establish nominal 95% coverage."
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_and_solution() -> Result<()> {
        report(interval, calibration)?;
        crate::solutions::ch12::check()
    }
}
