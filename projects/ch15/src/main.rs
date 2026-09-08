//! Small decision tree, random forest, and squared-error gradient boosting.
#[derive(Clone, Copy, Debug)]
struct Row {
    features: [f64; 2],
    label: usize,
    target: f64,
}
#[derive(Debug)]
enum Tree {
    Leaf(usize),
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Tree>,
        right: Box<Tree>,
    },
}

fn majority(rows: &[Row]) -> usize {
    let ones = rows.iter().filter(|r| r.label == 1).count();
    (ones * 2 >= rows.len()) as usize
}
fn gini_impurity(rows: &[Row]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    let p = rows.iter().filter(|r| r.label == 1).count() as f64 / rows.len() as f64;
    2.0 * p * (1.0 - p)
}
fn weighted_impurity(left: &[Row], right: &[Row]) -> f64 {
    let total = (left.len() + right.len()) as f64;
    (left.len() as f64 * gini_impurity(left) + right.len() as f64 * gini_impurity(right)) / total
}
fn fit_tree(rows: &[Row], depth: usize, max_depth: usize, features: &[usize]) -> Tree {
    if depth == max_depth || rows.len() < 2 || gini_impurity(rows) == 0.0 {
        return Tree::Leaf(majority(rows));
    }
    let mut best: Option<(f64, usize, f64)> = None;
    for &j in features {
        for row in rows {
            let threshold = row.features[j];
            let (left, right): (Vec<Row>, Vec<Row>) = rows
                .iter()
                .copied()
                .partition(|row| row.features[j] <= threshold);
            if left.is_empty() || right.is_empty() {
                continue;
            }
            let impurity = weighted_impurity(&left, &right);
            if best.is_none_or(|best| impurity < best.0) {
                best = Some((impurity, j, threshold));
            }
        }
    }
    let Some((_, j, threshold)) = best else {
        return Tree::Leaf(majority(rows));
    };
    let (left, right): (Vec<Row>, Vec<Row>) = rows
        .iter()
        .copied()
        .partition(|row| row.features[j] <= threshold);
    Tree::Split {
        feature: j,
        threshold,
        left: Box::new(fit_tree(&left, depth + 1, max_depth, features)),
        right: Box::new(fit_tree(&right, depth + 1, max_depth, features)),
    }
}
impl Tree {
    fn predict(&self, features: [f64; 2]) -> usize {
        match self {
            Tree::Leaf(c) => *c,
            Tree::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if features[*feature] <= *threshold {
                    left.predict(features)
                } else {
                    right.predict(features)
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn index(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }
}
fn fit_random_tree(rows: &[Row], depth: usize, max_depth: usize, rng: &mut Rng) -> Tree {
    if depth == max_depth || rows.len() < 2 || gini_impurity(rows) == 0.0 {
        return Tree::Leaf(majority(rows));
    }
    let feature = rng.index(2);
    let mut best: Option<(f64, f64)> = None;
    for r in rows {
        let threshold = r.features[feature];
        let (left, right): (Vec<Row>, Vec<Row>) = rows
            .iter()
            .copied()
            .partition(|row| row.features[feature] <= threshold);
        if left.is_empty() || right.is_empty() {
            continue;
        }
        let impurity = weighted_impurity(&left, &right);
        if best.is_none_or(|best| impurity < best.0) {
            best = Some((impurity, threshold));
        }
    }
    let Some((_, threshold)) = best else {
        return Tree::Leaf(majority(rows));
    };
    let (left, right): (Vec<Row>, Vec<Row>) = rows
        .iter()
        .copied()
        .partition(|row| row.features[feature] <= threshold);
    Tree::Split {
        feature,
        threshold,
        left: Box::new(fit_random_tree(&left, depth + 1, max_depth, rng)),
        right: Box::new(fit_random_tree(&right, depth + 1, max_depth, rng)),
    }
}
fn fit_forest(rows: &[Row], trees: usize, seed: u64) -> Result<Vec<Tree>, &'static str> {
    if rows.is_empty() || trees == 0 {
        return Err("forest needs rows and trees");
    };
    let mut rng = Rng(seed.max(1));
    let mut out = Vec::new();
    for _ in 0..trees {
        let sample: Vec<Row> = (0..rows.len())
            .map(|_| rows[rng.index(rows.len())])
            .collect();
        out.push(fit_random_tree(&sample, 0, 3, &mut rng));
    }
    Ok(out)
}
fn forest_predict(trees: &[Tree], features: [f64; 2]) -> usize {
    let ones = trees
        .iter()
        .filter(|tree| tree.predict(features) == 1)
        .count();
    (ones * 2 >= trees.len()) as usize
}

#[derive(Clone, Copy, Debug)]
struct Stump {
    feature: usize,
    threshold: f64,
    left: f64,
    right: f64,
}
impl Stump {
    fn predict(self, features: [f64; 2]) -> f64 {
        if features[self.feature] <= self.threshold {
            self.left
        } else {
            self.right
        }
    }
}
fn fit_stump(rows: &[Row], residuals: &[f64]) -> Stump {
    let mut best = (
        f64::INFINITY,
        Stump {
            feature: 0,
            threshold: rows[0].features[0],
            left: 0.0,
            right: 0.0,
        },
    );
    for j in 0..2 {
        for row in rows {
            let threshold = row.features[j];
            let mut ls = 0.;
            let mut ln = 0.;
            let mut rs = 0.;
            let mut rn = 0.;
            for (row, &e) in rows.iter().zip(residuals) {
                if row.features[j] <= threshold {
                    ls += e;
                    ln += 1.
                } else {
                    rs += e;
                    rn += 1.
                }
            }
            if ln == 0. || rn == 0. {
                continue;
            }
            let lm = ls / ln;
            let rm = rs / rn;
            let residual_sse = rows
                .iter()
                .zip(residuals)
                .map(|(row, residual)| {
                    (residual - if row.features[j] <= threshold { lm } else { rm }).powi(2)
                })
                .sum();
            if residual_sse < best.0 {
                best = (
                    residual_sse,
                    Stump {
                        feature: j,
                        threshold,
                        left: lm,
                        right: rm,
                    },
                );
            }
        }
    }
    best.1
}
#[derive(Debug)]
struct Boost {
    base: f64,
    learning_rate: f64,
    stumps: Vec<Stump>,
}
impl Boost {
    fn fit(rows: &[Row], rounds: usize, learning_rate: f64) -> Result<Self, &'static str> {
        if rows.is_empty()
            || rounds == 0
            || !(0.0..=1.0).contains(&learning_rate)
            || learning_rate == 0.0
        {
            return Err("boosting needs rows, rounds, and a rate in (0,1]");
        };
        let base = rows.iter().map(|r| r.target).sum::<f64>() / rows.len() as f64;
        let mut predictions = vec![base; rows.len()];
        let mut stumps = Vec::new();
        for _ in 0..rounds {
            let residuals: Vec<f64> = rows
                .iter()
                .zip(&predictions)
                .map(|(row, prediction)| row.target - prediction)
                .collect();
            let stump = fit_stump(rows, &residuals);
            for (prediction, row) in predictions.iter_mut().zip(rows) {
                *prediction += learning_rate * stump.predict(row.features);
            }
            stumps.push(stump)
        }
        Ok(Self {
            base,
            learning_rate,
            stumps,
        })
    }
    fn predict(&self, features: [f64; 2]) -> f64 {
        self.base
            + self.learning_rate
                * self
                    .stumps
                    .iter()
                    .map(|stump| stump.predict(features))
                    .sum::<f64>()
    }
}

fn main() -> Result<(), &'static str> {
    let rows = [
        Row {
            features: [0., 0.],
            label: 0,
            target: 0.,
        },
        Row {
            features: [1., 0.2],
            label: 0,
            target: 1.,
        },
        Row {
            features: [2., 0.1],
            label: 0,
            target: 2.,
        },
        Row {
            features: [3., 1.],
            label: 1,
            target: 3.,
        },
        Row {
            features: [4., 0.8],
            label: 1,
            target: 4.,
        },
        Row {
            features: [5., 1.2],
            label: 1,
            target: 5.,
        },
    ];
    let tree = fit_tree(&rows, 0, 3, &[0, 1]);
    let trees = fit_forest(&rows, 31, 9)?;
    let boost = Boost::fit(&rows, 20, 0.2)?;
    println!(
        "tree class {}, forest class {}",
        tree.predict([3.5, 0.9]),
        forest_predict(&trees, [3.5, 0.9])
    );
    println!(
        "boosted regression prediction {:.3}",
        boost.predict([3.5, 0.9])
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_three_ensembles_work() {
        let d = [
            Row {
                features: [0., 0.],
                label: 0,
                target: 0.,
            },
            Row {
                features: [1., 0.],
                label: 0,
                target: 1.,
            },
            Row {
                features: [3., 1.],
                label: 1,
                target: 3.,
            },
            Row {
                features: [4., 1.],
                label: 1,
                target: 4.,
            },
        ];
        assert_eq!(fit_tree(&d, 0, 2, &[0, 1]).predict([3.5, 1.]), 1);
        assert_eq!(
            forest_predict(&fit_forest(&d, 51, 2).unwrap(), [3.5, 1.]),
            1
        );
        assert_eq!(fit_tree(&d, 0, 2, &[0, 1]).predict([0.5, 0.]), 0);
        assert_eq!(
            forest_predict(&fit_forest(&d, 51, 2).unwrap(), [0.5, 0.]),
            0
        );
        assert!((gini_impurity(&d) - 0.5).abs() < 1e-12);
        assert_eq!(weighted_impurity(&d[..2], &d[2..]), 0.0);
        let b = Boost::fit(&d, 40, 0.2).unwrap();
        assert!((b.predict([4., 1.]) - 4.).abs() < 0.05);
    }
}
