//! Learner: replace row splitting and a fixed cutoff with a grouped evaluation protocol.
use crate::classifier;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Record {
    pub machine: usize,
    pub features: [f64; 2],
    pub fault: bool,
    pub after_inspection: bool,
}
pub type Splits = [Vec<Record>; 3];
pub fn records() -> Vec<Record> {
    (0..60)
        .flat_map(|machine| {
            (0..2).map(move |visit| {
                let a = ((machine * 17) % 31) as f64 / 10. - 1.5;
                let b = ((machine * 7) % 19) as f64 / 10. - 0.9;
                let noise = if machine % 11 == 0 { 1.4 } else { 0. };
                let fault = (a + 0.7 * b + noise > 1.) ^ (machine % 13 == 0);
                Record {
                    machine,
                    features: [a + visit as f64 * 0.08, b],
                    fault,
                    after_inspection: fault,
                }
            })
        })
        .collect()
}
pub fn split(rows: &[Record]) -> Result<Splits, String> {
    validate_records(rows)?;
    let mut result: [Vec<Record>; 3] = std::array::from_fn(|_| Vec::new());
    // Working comparison: a row split; run reports its repeated-machine leakage.
    for (i, row) in rows.iter().enumerate() {
        let part = match i % 5 {
            0 => 1,
            1 => 2,
            _ => 0,
        };
        result[part].push(*row);
    }
    Ok(result)
}
pub fn choose_threshold(
    validation: &[(f64, bool)],
    candidates: &[f64],
    miss_cost: usize,
) -> Result<f64, String> {
    let _ = miss_cost;
    validate_scores(validation, 0.5)?;
    if candidates.is_empty() {
        return Err("provide threshold candidates".into());
    }
    for &t in candidates {
        validate_scores(validation, t)?;
    }
    Ok(0.5)
}
pub fn validate_records(rows: &[Record]) -> Result<(), String> {
    if rows.is_empty()
        || rows
            .iter()
            .any(|r| r.features.iter().any(|v| !v.is_finite()))
    {
        Err("records need finite features and at least one row".into())
    } else {
        Ok(())
    }
}
pub fn validate_scores(rows: &[(f64, bool)], t: f64) -> Result<(), String> {
    if rows.is_empty()
        || !t.is_finite()
        || !(0.0..=1.0).contains(&t)
        || rows
            .iter()
            .any(|(p, _)| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        Err("nonempty probabilities and threshold in [0,1] required".into())
    } else {
        Ok(())
    }
}
#[derive(Debug, Default, PartialEq)]
pub struct Counts {
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_: usize,
}
pub fn evaluate(rows: &[(f64, bool)], threshold: f64) -> Result<Counts, String> {
    validate_scores(rows, threshold)?;
    let mut c = Counts::default();
    for &(p, y) in rows {
        match (p >= threshold, y) {
            (true, true) => c.tp += 1,
            (true, false) => c.fp += 1,
            (false, false) => c.tn += 1,
            (false, true) => c.fn_ += 1,
        }
    }
    Ok(c)
}
fn ratio(n: usize, d: usize) -> f64 {
    if d == 0 {
        0.
    } else {
        n as f64 / d as f64
    }
}
impl Counts {
    pub fn cost(&self, miss_cost: usize) -> usize {
        miss_cost * self.fn_ + self.fp
    }
    pub fn print(&self, miss_cost: usize) {
        println!("{self:?}; accuracy={:.3}, precision={:.3}, recall={:.3}, F1={:.3}, balanced accuracy={:.3}, cost({miss_cost}FN+FP)={}",ratio(self.tp+self.tn,self.tp+self.fp+self.tn+self.fn_),ratio(self.tp,self.tp+self.fp),ratio(self.tp,self.tp+self.fn_),ratio(2*self.tp,2*self.tp+self.fp+self.fn_),0.5*(ratio(self.tp,self.tp+self.fn_)+ratio(self.tn,self.tn+self.fp)),self.cost(miss_cost));
    }
}
pub fn disjoint(splits: &Splits) -> bool {
    (0..3).all(|i| {
        (i + 1..3).all(|j| {
            let ids: std::collections::HashSet<_> = splits[i].iter().map(|r| r.machine).collect();
            splits[j].iter().all(|r| !ids.contains(&r.machine))
        })
    })
}
pub type Splitter = fn(&[Record]) -> Result<Splits, String>;
pub type Selector = fn(&[(f64, bool)], &[f64], usize) -> Result<f64, String>;
pub fn report(split: Splitter, select: Selector, args: &[String]) -> Result<(), String> {
    let o = crate::args::Options::parse(args, &["--miss-cost", "--candidate", "--variant"])?;
    let miss_cost = o.count("--miss-cost", 5)?;
    let mut rows = records();
    match o.text("--variant", "original") {
        "original" => {}
        "extra-visit" => {
            let mut extra = rows[0];
            extra.features[0] += 0.16;
            rows.push(extra);
            rows.reverse();
        }
        _ => return Err("variant must be original or extra-visit".into()),
    }
    let groups = split(&rows)?;
    let clean = disjoint(&groups);
    println!(
        "split sizes={:?}; machine-disjoint={clean}",
        groups.each_ref().map(|x| x.len())
    );
    if !clean {
        println!("LEAKAGE: repeated machines cross splits; following numbers are a contaminated demonstration, not a final estimate.");
    }
    let train: Vec<_> = groups[0].iter().map(|r| (r.features, r.fault)).collect();
    let model = crate::solutions::ch04::train(&train, 500, 0.15)?;
    let scores = |data: &[Record]| {
        data.iter()
            .map(|r| {
                (
                    classifier::sigmoid(classifier::logit(model, r.features)),
                    r.fault,
                )
            })
            .collect::<Vec<_>>()
    };
    let val = scores(&groups[1]);
    let mut candidates = vec![0.2, 0.35, 0.5, 0.65, 0.8];
    let extra = o.number("--candidate", 0.5)?;
    if !candidates.contains(&extra) {
        candidates.push(extra);
    }
    println!("validation candidates={candidates:?}; miss cost={miss_cost}, false-alarm cost=1");
    let threshold = select(&val, &candidates, miss_cost)?;
    println!("threshold frozen from validation={threshold}; test candidate:");
    evaluate(&scores(&groups[2]), threshold)?.print(miss_cost);
    for (row, (p, y)) in groups[2].iter().zip(scores(&groups[2])) {
        if (p >= threshold) != y {
            println!(
                "residual error: machine={}, features={:?}, p={p:.4}, actual fault={y}",
                row.machine, row.features
            );
        }
    }
    let majority: Vec<_> = groups[2].iter().map(|r| (0., r.fault)).collect();
    println!("always-negative on same test rows:");
    evaluate(&majority, 0.5)?.print(miss_cost);
    let leaked: Vec<_> = groups[2]
        .iter()
        .map(|r| (f64::from(r.after_inspection), r.fault))
        .collect();
    println!("forbidden after-inspection feature (available only AFTER outcome):");
    evaluate(&leaked, 0.5)?.print(miss_cost);
    Ok(())
}
pub fn verify(split: Splitter, select: Selector) -> Result<(), String> {
    let rows = records();
    let parts = split(&rows)?;
    if !disjoint(&parts) {
        return Err("GOAL_NOT_MET: split complete machines, not their individual visits".into());
    }
    if parts.iter().map(Vec::len).sum::<usize>() != rows.len() || parts.iter().any(Vec::is_empty) {
        return Err("GOAL_NOT_MET: split lost rows or an entire partition".into());
    }
    for row in &rows {
        if parts.iter().flatten().filter(|r| *r == row).count() != 1 {
            return Err("GOAL_NOT_MET: each source row must occur exactly once".into());
        }
    }
    let val = [(0.72, true), (0.41, true), (0.6, false), (0.2, false)];
    let t = select(&val, &[0.3, 0.5, 0.7], 5)?;
    println!("changed validation set: chosen threshold={t}, expected0.3 under5FN+FP");
    if t != 0.3 {
        return Err(
            "GOAL_NOT_MET: select minimum validation cost, first candidate wins ties".into(),
        );
    }
    if select(&[(0.9, true), (0.6, false)], &[0.3, 0.7], 5)? != 0.7
        || select(&[(0.9, true), (0.1, false)], &[0.7, 0.3], 5)? != 0.7
    {
        return Err(
            "GOAL_NOT_MET: selection must adapt to scores and preserve first-candidate ties".into(),
        );
    }
    let reversed: Vec<_> = rows.iter().copied().rev().collect();
    let reversed = split(&reversed)?;
    for i in 0..3 {
        let a: std::collections::HashSet<_> = parts[i].iter().map(|r| r.machine).collect();
        let b: std::collections::HashSet<_> = reversed[i].iter().map(|r| r.machine).collect();
        if a != b {
            return Err("GOAL_NOT_MET: reordering rows changed machine assignments".into());
        }
    }
    Ok(())
}
pub fn run(args: &[String]) -> Result<(), String> {
    report(split, choose_threshold, args)
}
pub fn check() -> Result<(), String> {
    verify(split, choose_threshold)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learner_baseline_counts_and_solution() -> Result<(), String> {
        let p = split(&records())?;
        assert_eq!(p.iter().map(Vec::len).sum::<usize>(), 120);
        assert!(choose_threshold(&[(0.2, false)], &[0.3], 5).is_ok());
        let c = evaluate(
            &[(0.2, false), (0.45, true), (0.55, false), (0.9, true)],
            0.5,
        )?;
        assert_eq!(
            c,
            Counts {
                tp: 1,
                fp: 1,
                tn: 1,
                fn_: 1
            }
        );
        assert!(evaluate(&[(f64::NAN, true)], 0.5).is_err());
        assert!(split(&[]).is_err());
        crate::solutions::ch05::check()
    }
}
