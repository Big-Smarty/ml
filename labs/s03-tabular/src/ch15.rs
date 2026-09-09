//! Learner: replace a median stump with recursive CART, independent row/feature
//! randomization with a forest, and a global mean with residual boosting.
use crate::data::{self, Point, Result};
#[derive(Clone, Debug)]
pub enum Tree {
    Leaf(f64),
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Tree>,
        right: Box<Tree>,
    },
}
impl Tree {
    pub fn predict(&self, x: &[f64]) -> f64 {
        match self {
            Self::Leaf(p) => *p,
            Self::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if x[*feature] <= *threshold {
                    left.predict(x)
                } else {
                    right.predict(x)
                }
            }
        }
    }
    pub fn nodes(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Split { left, right, .. } => 1 + left.nodes() + right.nodes(),
        }
    }
}
pub fn mean_label(rows: &[Point]) -> f64 {
    rows.iter().map(|r| f64::from(r.label)).sum::<f64>() / rows.len() as f64
}
pub fn gini(rows: &[Point]) -> f64 {
    if rows.is_empty() {
        0.
    } else {
        let p = mean_label(rows);
        2. * p * (1. - p)
    }
}
pub type TreeFit = fn(&[Point], usize) -> Result<Tree>;
pub type ForestFit = fn(&[Point], usize, usize, usize, u64) -> Result<Vec<Tree>>;
pub type BoostFit = fn(&[Point], &[f64], usize, f64) -> Result<Boost>;
#[derive(Debug)]
pub struct Boost {
    pub base: f64,
    pub rate: f64,
    pub stages: Vec<Tree>,
}
impl Boost {
    pub fn predict(&self, x: &[f64]) -> f64 {
        self.base + self.rate * self.stages.iter().map(|t| t.predict(x)).sum::<f64>()
    }
}
pub fn tree(rows: &[Point], depth: usize) -> Result<Tree> {
    data::validate_points(rows)?;
    if depth > 32 {
        return Err("depth must be at most32".into());
    }
    if depth == 0 {
        return Ok(Tree::Leaf(mean_label(rows)));
    }
    let mut values: Vec<_> = rows.iter().map(|r| r.features[0]).collect();
    values.sort_by(f64::total_cmp);
    let threshold = values[values.len() / 2];
    let (left, right): (Vec<_>, Vec<_>) = rows
        .iter()
        .cloned()
        .partition(|r| r.features[0] <= threshold);
    if left.is_empty() || right.is_empty() {
        return Ok(Tree::Leaf(mean_label(rows)));
    }
    Ok(Tree::Split {
        feature: 0,
        threshold,
        left: Box::new(Tree::Leaf(mean_label(&left))),
        right: Box::new(Tree::Leaf(mean_label(&right))),
    })
}
pub fn forest(
    rows: &[Point],
    count: usize,
    depth: usize,
    mtry: usize,
    _seed: u64,
) -> Result<Vec<Tree>> {
    let d = data::validate_points(rows)?;
    if count == 0 || count > 100 || mtry == 0 || mtry > d {
        return Err("need1..100 trees and1..d feature candidates".into());
    }
    // Repeated identical stumps are an intact baseline, but they provide no diversity.
    Ok(vec![tree(rows, depth)?; count])
}
pub fn boost(rows: &[Point], targets: &[f64], rounds: usize, rate: f64) -> Result<Boost> {
    initial_boost(rows, targets, rounds, rate)
}
pub fn initial_boost(rows: &[Point], targets: &[f64], rounds: usize, rate: f64) -> Result<Boost> {
    data::validate_points(rows)?;
    if targets.len() != rows.len()
        || targets.iter().any(|v| !v.is_finite())
        || rounds == 0
        || rounds > 200
        || !rate.is_finite()
        || rate <= 0.
        || rate > 1.
    {
        return Err("boosting needs paired finite targets,1..200 rounds and rate in(0,1]".into());
    }
    Ok(Boost {
        base: data::finite(targets.iter().sum::<f64>() / targets.len() as f64)?,
        rate,
        stages: vec![],
    })
}
pub fn forest_predict(trees: &[Tree], x: &[f64]) -> f64 {
    trees.iter().map(|t| t.predict(x)).sum::<f64>() / trees.len() as f64
}
pub fn run(_: &[String]) -> Result<()> {
    report(tree, forest, boost)
}
pub fn check() -> Result<()> {
    verify(tree, forest, boost)
}
pub fn report(tree: TreeFit, forest: ForestFit, boost: BoostFit) -> Result<()> {
    let (raw, train, valid) = crate::ch14::prepared()?;
    for depth in [1, 3, 6] {
        let t = tree(&train, depth)?;
        let p: Vec<_> = valid.iter().map(|r| t.predict(&r.features)).collect();
        crate::evaluation::show(
            &format!("tree depth{depth} nodes{}", t.nodes()),
            &data::predictions(&raw, &p)?,
        )?;
        println!(
            " train Gini {:.4}; train accuracy {:.3}",
            gini(&train),
            train
                .iter()
                .filter(|r| (t.predict(&r.features) >= 0.5) == (r.label == 1))
                .count() as f64
                / train.len() as f64
        );
    }
    for mtry in [train[0].features.len(), 3] {
        let trees = forest(&train, 15, 4, mtry, 7)?;
        let p: Vec<_> = valid
            .iter()
            .map(|r| forest_predict(&trees, &r.features))
            .collect();
        crate::evaluation::show(
            if mtry == 3 {
                "random forest"
            } else {
                "bagging(all features)"
            },
            &data::predictions(&raw, &p)?,
        )?;
    }
    let targets: Vec<_> = train.iter().map(|r| f64::from(r.label)).collect();
    for rounds in [1, 10, 30] {
        let model = boost(&train, &targets, rounds, 0.2)?;
        let p: Vec<_> = valid
            .iter()
            .map(|r| model.predict(&r.features).clamp(0., 1.))
            .collect();
        crate::evaluation::show(
            &format!("squared-error boosting rounds{rounds}; clipped0/1 regression"),
            &data::predictions(&raw, &p)?,
        )?;
    }
    Ok(())
}
pub fn verify(tree: TreeFit, forest: ForestFit, boost: BoostFit) -> Result<()> {
    let rows: Vec<_> = [(-1., -1., 0), (-1., 1., 1), (1., -1., 1), (1., 1., 0)]
        .into_iter()
        .flat_map(|(a, b, label)| {
            [
                Point {
                    features: vec![a, b],
                    label,
                },
                Point {
                    features: vec![a, b],
                    label,
                },
            ]
        })
        .collect();
    let t = tree(&rows, 2)?;
    if rows
        .iter()
        .any(|r| (t.predict(&r.features) >= 0.5) != (r.label == 1))
    {
        return data::goal("Goal not met: a depth2 tree must learn the two-feature interaction; search weighted impurity then recurse".into());
    }
    let sample: Vec<_> = (0..16)
        .map(|i| Point {
            features: vec![i as f64, (i % 3) as f64],
            label: u8::from(i >= 8),
        })
        .collect();
    let trees = forest(&sample, 21, 3, 1, 11)?;
    let structures: std::collections::BTreeSet<_> =
        trees.iter().map(|t| format!("{t:?}")).collect();
    if structures.len() < 2
        || forest_predict(&trees, &[14., 2.]) <= forest_predict(&trees, &[1., 1.])
    {
        return data::goal("Goal not met: bootstrap independent trees and choose fresh candidate features at every node".into());
    }
    let regression: Vec<_> = (0..4)
        .map(|i| Point {
            features: vec![i as f64],
            label: 0,
        })
        .collect();
    let y = [-2., -1., 2., 4.];
    let model = boost(&regression, &y, 80, 0.2)?;
    let mse = regression
        .iter()
        .zip(y)
        .map(|(r, y)| (model.predict(&r.features) - y).powi(2))
        .sum::<f64>()
        / 4.;
    if mse > 0.005 || model.stages.len() != 80 {
        return data::goal(format!("Goal not met: fit each stump to target-current_prediction; changed regression MSE={mse:.4}"));
    }
    let constant = vec![
        Point {
            features: vec![0.],
            label: 0
        };
        3
    ];
    let c = boost(&constant, &[1., 2., 6.], 10, 0.2)?;
    if (c.predict(&[0.]) - 3.).abs() > 1e-12 {
        return data::goal("constant features must retain the mean, not invent a split".into());
    }
    println!("Recursive interaction, randomized structure and negative/positive residual-target checks passed; MSE={mse:.6}");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_and_solution() -> Result<()> {
        let (_, train, _) = crate::ch14::prepared()?;
        assert!(tree(&train, 3)?.predict(&train[0].features).is_finite());
        assert!(forest(&train, 0, 3, 1, 7).is_err());
        crate::solutions::ch15::check()
    }
}
