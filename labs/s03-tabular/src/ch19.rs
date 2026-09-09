//! Learner: replace one fixed holdout and first-candidate selection with the
//! frozen group/time CV procedure. Reuse completed prior models; do not retune on final rows.
use crate::data::{self, Prediction, RawRow, Result};
use crate::evaluation::{self, Metrics};
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Config {
    Knn(usize),
    Bayes(f64),
    Tree(usize),
}
impl Config {
    pub fn family(self) -> usize {
        match self {
            Self::Knn(_) => 0,
            Self::Bayes(_) => 1,
            Self::Tree(_) => 2,
        }
    }
}
pub const CANDIDATES: [Config; 6] = [
    Config::Knn(3),
    Config::Knn(7),
    Config::Bayes(0.02),
    Config::Bayes(0.2),
    Config::Tree(2),
    Config::Tree(4),
];
#[derive(Clone, Debug)]
pub struct Fold {
    pub train: Vec<RawRow>,
    pub valid: Vec<RawRow>,
}
#[derive(Clone, Debug)]
pub struct Trial {
    pub config: Config,
    pub fold_costs: Vec<f64>,
    pub predictions: Vec<Prediction>,
    pub metrics: Metrics,
}
pub type Folds = fn(&[RawRow]) -> Result<Vec<Fold>>;
pub type Search = fn(&[Fold], &[Config]) -> Result<(usize, Vec<Trial>)>;
pub fn folds(rows: &[RawRow]) -> Result<Vec<Fold>> {
    let (train, valid) = crate::solutions::ch13::split(rows, crate::ch13::Split::FutureGroup)?;
    // Working single-holdout checkpoint. Each machine needs a turn as future validation.
    Ok(vec![Fold { train, valid }])
}
pub fn fitted_predictions(
    train: &[RawRow],
    valid: &[RawRow],
    config: Config,
) -> Result<Vec<Prediction>> {
    let prep = crate::solutions::ch13::fit(train)?;
    let x = prep.transform_all(train)?;
    let v = prep.transform_all(valid)?;
    let probabilities = match config {
        Config::Knn(k) => v
            .iter()
            .map(|r| crate::solutions::ch14::knn(&x, &r.features, k))
            .collect::<Result<Vec<_>>>()?,
        Config::Bayes(floor) => {
            let m = crate::solutions::ch14::bayes(&x, floor)?;
            v.iter()
                .map(|r| m.probability(&r.features))
                .collect::<Result<Vec<_>>>()?
        }
        Config::Tree(depth) => {
            let m = crate::solutions::ch15::tree(&x, depth)?;
            v.iter().map(|r| m.predict(&r.features)).collect()
        }
    };
    data::predictions(valid, &probabilities)
}
pub fn trial(folds: &[Fold], config: Config) -> Result<Trial> {
    if folds.is_empty() {
        return Err("need development folds".into());
    }
    let mut predictions = Vec::new();
    let mut fold_costs = Vec::new();
    for f in folds {
        let p = fitted_predictions(&f.train, &f.valid, config)?;
        fold_costs.push(evaluation::metrics(&p)?.cost());
        predictions.extend(p);
    }
    let metrics = evaluation::metrics(&predictions)?;
    Ok(Trial {
        config,
        fold_costs,
        predictions,
        metrics,
    })
}
pub fn search(folds: &[Fold], candidates: &[Config]) -> Result<(usize, Vec<Trial>)> {
    let &config = candidates.first().ok_or("need candidates")?;
    // First-candidate checkpoint; no hyperparameter search is being claimed.
    Ok((0, vec![trial(folds, config)?]))
}
pub fn run(args: &[String]) -> Result<()> {
    report(folds, search, args.iter().any(|a| a == "--final"))
}
pub fn check() -> Result<()> {
    verify(folds, search)
}
pub fn report(folds: Folds, search: Search, open_final: bool) -> Result<()> {
    let rows = data::load()?;
    let dev = data::development(&rows);
    let fs = folds(&dev)?;
    println!("maintenance-v1 FNV1a64={:016x}; split=early train<=day3; validation days5..7 on held machines; final machines18..23 days9..11",data::fingerprint());
    println!("Frozen metric=(4FN+FP)/n, threshold=.5, lower is better; candidate order={CANDIDATES:?}; no test-dependent choices.");
    for (i, f) in fs.iter().enumerate() {
        println!(
            "fold{i} train IDs={:?}; valid IDs={:?}",
            f.train.iter().map(|r| r.id).collect::<Vec<_>>(),
            f.valid.iter().map(|r| r.id).collect::<Vec<_>>()
        );
    }
    let (best, trials) = search(&fs, &CANDIDATES)?;
    for (i, t) in trials.iter().enumerate() {
        println!(
            "candidate{i} {:?}: fold costs={:?}; pooled cost={:.4}; accuracy={:.4}; Brier={:.4}",
            t.config,
            t.fold_costs,
            t.metrics.cost(),
            t.metrics.accuracy(),
            t.metrics.brier
        );
    }
    let winner = trials.get(best).ok_or("selected index is invalid")?.config;
    println!("Frozen development choice: {winner:?}");
    if !open_final {
        println!("Final rows remain unopened. After recording the protocol and selection, rerun19 --final (or19 --solution --final). Inspecting final errors begins a new development cycle.");
        return Ok(());
    }
    if trials.len() != CANDIDATES.len() || fs.len() != 3 {
        return Err("Final report requires the complete three-fold, six-candidate protocol; finish the learner functions or inspect --solution".into());
    }
    final_report(&dev, &data::final_test(&rows), &trials, winner)
}
pub fn finalists(trials: &[Trial]) -> Result<Vec<Config>> {
    let mut result = Vec::new();
    for family in 0..3 {
        let best = trials
            .iter()
            .filter(|t| t.config.family() == family)
            .min_by(|a, b| a.metrics.cost().total_cmp(&b.metrics.cost()))
            .ok_or("missing model family")?;
        result.push(best.config);
    }
    Ok(result)
}
pub fn final_report(
    dev: &[RawRow],
    test: &[RawRow],
    trials: &[Trial],
    winner: Config,
) -> Result<()> {
    let prevalence = dev.iter().map(|r| f64::from(r.label)).sum::<f64>() / dev.len() as f64;
    let majority = f64::from(prevalence >= 0.5);
    evaluation::show(
        "majority final",
        &data::predictions(test, &vec![majority; test.len()])?,
    )?;
    evaluation::show(
        "training-prevalence final",
        &data::predictions(test, &vec![prevalence; test.len()])?,
    )?;
    for config in finalists(trials)? {
        let p = fitted_predictions(dev, test, config)?;
        evaluation::show(
            &format!("FINAL {config:?} selected={}", config == winner),
            &p,
        )?;
        let interval = crate::solutions::ch12::interval(&p, 2000, 7)?;
        println!(
            " cluster-bootstrap accuracy interval={interval:?}; machines=6 (coverage fragile)"
        );
        for (name, missing, turbo) in [
            ("temperature missing", true, false),
            ("temperature observed", false, false),
            ("new turbo regime", false, true),
        ] {
            let slice: Vec<_> = test
                .iter()
                .zip(&p)
                .filter(|(r, _)| {
                    if turbo {
                        r.regime == "turbo"
                    } else {
                        r.sensors[0].is_none() == missing
                    }
                })
                .map(|(_, p)| *p)
                .collect();
            if !slice.is_empty() {
                evaluation::show(name, &slice)?;
            }
        }
        for (r, p) in test
            .iter()
            .zip(&p)
            .filter(|(_, p)| (p.probability >= 0.5) != (p.label == 1))
        {
            println!(" error id={} machine={} day={} regime={} missing_temp={} p={:.3} label={} signed_probability_error={:.3}",p.id,r.machine,r.day,r.regime,r.sensors[0].is_none(),p.probability,p.label,p.probability-f64::from(p.label));
        }
    }
    println!("Interpretation: correlated/noisy sensors omit latent wear, Bernoulli labels overlap, later sensors drift+5 degrees, and turbo is unseen. Slices describe this18-row holdout; they do not prove causes or authorize retuning on it.");
    Ok(())
}
pub fn verify(folds: Folds, search: Search) -> Result<()> {
    let rows = data::load()?;
    let dev = data::development(&rows);
    let fs = folds(&dev)?;
    if fs.len() != 3 {
        return data::goal("Goal not met: build three rotated machine-group folds, each with earlier training and later held-machine validation".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    for f in &fs {
        if f.train.len() != 48
            || f.valid.len() != 18
            || f.train
                .iter()
                .any(|a| a.day > 3 || f.valid.iter().any(|b| a.machine == b.machine))
            || f.valid
                .iter()
                .any(|b| b.day < 5 || b.day > 7 || !seen.insert(b.id))
        {
            return data::goal("fold identities, chronology, group independence, or one-validation-prediction invariant failed".into());
        }
    }
    let (best, trials) = search(&fs, &CANDIDATES)?;
    if best >= trials.len()
        || trials.len() != 6
        || trials
            .iter()
            .zip(CANDIDATES)
            .any(|(t, c)| t.config != c || t.predictions.len() != 54)
    {
        return data::goal("Goal not met: evaluate every frozen candidate on every fold with newly fitted preprocessing".into());
    }
    if trials
        .iter()
        .any(|t| t.metrics.cost() < trials[best].metrics.cost())
    {
        return data::goal("select minimum pooled4FN+FP cost; retain earliest ties".into());
    }
    // Inspectable independent variation: reverse row order, retaining stable identities.
    let mut reversed = dev.clone();
    reversed.reverse();
    let reversed_folds = folds(&reversed)?;
    for (a, b) in fs.iter().zip(&reversed_folds) {
        if a.train.iter().map(|r| r.id).collect::<Vec<_>>()
            != b.train.iter().map(|r| r.id).collect::<Vec<_>>()
            || a.valid.iter().map(|r| r.id).collect::<Vec<_>>()
                != b.valid.iter().map(|r| r.id).collect::<Vec<_>>()
        {
            return data::goal("sort stable IDs so reordering source records preserves model traversal and membership".into());
        }
    }
    println!("Development protocol, six-candidate search and stable-order checks passed. Final outcomes remain unopened; use --final only after freezing selection.");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_and_solution() -> Result<()> {
        let d = data::development(&data::load()?);
        let (i, t) = search(&folds(&d)?, &CANDIDATES)?;
        assert_eq!(i, 0);
        assert_eq!(t.len(), 1);
        assert!(search(&[], &[]).is_err());
        crate::solutions::ch19::check()
    }
}
