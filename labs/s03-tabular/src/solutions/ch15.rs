//! CART minimizes weighted child Gini. Bagging resamples rows; forests also
//! redraw feature candidates inside grow. Boosting instead fits residuals in order.
use crate::ch15::{self, gini, mean_label, Boost, Tree};
use crate::data::{self, Point, Result, Rng};
fn grow(rows: &[Point], depth: usize, mtry: usize, rng: &mut Rng) -> Tree {
    if depth == 0 || rows.len() < 2 || gini(rows) == 0. {
        return Tree::Leaf(mean_label(rows));
    }
    let mut features: Vec<_> = (0..rows[0].features.len()).collect();
    if mtry < features.len() {
        for i in 0..mtry {
            let j = i + rng.index(features.len() - i);
            features.swap(i, j);
        }
        features.truncate(mtry);
    }
    let mut best: Option<(f64, usize, f64)> = None;
    // ponytail: scanning candidate partitions is quadratic in rows; sorted prefix counts for larger tables.
    for j in features {
        let mut values: Vec<_> = rows.iter().map(|r| r.features[j]).collect();
        values.sort_by(f64::total_cmp);
        values.dedup();
        for pair in values.windows(2) {
            let threshold = pair[0].midpoint(pair[1]);
            let (left, right): (Vec<_>, Vec<_>) = rows
                .iter()
                .cloned()
                .partition(|r| r.features[j] <= threshold);
            let score = (left.len() as f64 * gini(&left) + right.len() as f64 * gini(&right))
                / rows.len() as f64;
            if best.is_none_or(|b| score < b.0) {
                best = Some((score, j, threshold));
            }
        }
    }
    let Some((_, feature, threshold)) = best else {
        return Tree::Leaf(mean_label(rows));
    };
    let (left, right): (Vec<_>, Vec<_>) = rows
        .iter()
        .cloned()
        .partition(|r| r.features[feature] <= threshold);
    Tree::Split {
        feature,
        threshold,
        left: Box::new(grow(&left, depth - 1, mtry, rng)),
        right: Box::new(grow(&right, depth - 1, mtry, rng)),
    }
}
pub fn tree(rows: &[Point], depth: usize) -> Result<Tree> {
    let d = data::validate_points(rows)?;
    if depth > 32 {
        return Err("depth must be at most32".into());
    }
    Ok(grow(rows, depth, d, &mut Rng(7)))
}
pub fn forest(
    rows: &[Point],
    count: usize,
    depth: usize,
    mtry: usize,
    seed: u64,
) -> Result<Vec<Tree>> {
    let d = data::validate_points(rows)?;
    if count == 0 || count > 100 || mtry == 0 || mtry > d || depth > 32 {
        return Err("invalid forest settings".into());
    }
    let mut rng = Rng(seed);
    let mut trees = Vec::new();
    for _ in 0..count {
        let sample: Vec<_> = (0..rows.len())
            .map(|_| rows[rng.index(rows.len())].clone())
            .collect();
        trees.push(grow(&sample, depth, mtry, &mut rng));
    }
    Ok(trees)
}
fn stump(rows: &[Point], residuals: &[f64]) -> Tree {
    let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let mut best = (
        residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>(),
        Tree::Leaf(mean),
    );
    for j in 0..rows[0].features.len() {
        let mut values: Vec<_> = rows.iter().map(|r| r.features[j]).collect();
        values.sort_by(f64::total_cmp);
        values.dedup();
        for pair in values.windows(2) {
            let threshold = pair[0].midpoint(pair[1]);
            let mut sums = [0.; 2];
            let mut counts = [0; 2];
            for (row, &r) in rows.iter().zip(residuals) {
                let side = usize::from(row.features[j] > threshold);
                sums[side] += r;
                counts[side] += 1;
            }
            let means = [sums[0] / counts[0] as f64, sums[1] / counts[1] as f64];
            let sse = rows
                .iter()
                .zip(residuals)
                .map(|(row, r)| (r - means[usize::from(row.features[j] > threshold)]).powi(2))
                .sum();
            if sse < best.0 {
                best = (
                    sse,
                    Tree::Split {
                        feature: j,
                        threshold,
                        left: Box::new(Tree::Leaf(means[0])),
                        right: Box::new(Tree::Leaf(means[1])),
                    },
                );
            }
        }
    }
    best.1
}
pub fn boost(rows: &[Point], targets: &[f64], rounds: usize, rate: f64) -> Result<Boost> {
    let mut model = ch15::initial_boost(rows, targets, rounds, rate)?;
    let mut predictions = vec![model.base; rows.len()];
    for _ in 0..rounds {
        let residuals: Vec<_> = targets
            .iter()
            .zip(&predictions)
            .map(|(y, p)| y - p)
            .collect();
        let stage = stump(rows, &residuals);
        for (p, row) in predictions.iter_mut().zip(rows) {
            *p = data::finite(*p + rate * stage.predict(&row.features))?;
        }
        model.stages.push(stage);
    }
    Ok(model)
}
pub fn run(_: &[String]) -> Result<()> {
    ch15::report(tree, forest, boost)
}
pub fn check() -> Result<()> {
    ch15::verify(tree, forest, boost)
}
