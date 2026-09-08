//! Reproducible cross-validation and bounded search on a bundled tabular fixture.

#[derive(Clone, Copy, Debug)]
struct Row {
    id: u64,
    x: [f64; 2],
    y: u8,
}

fn validate(rows: &[Row]) -> Result<(), &'static str> {
    if rows.len() < 2 {
        return Err("evaluation requires at least two rows");
    }
    if rows
        .iter()
        .any(|r| r.y > 1 || r.x.iter().any(|x| !x.is_finite()))
    {
        return Err("features must be finite and labels must be 0 or 1");
    }
    let mut ids = std::collections::HashSet::new();
    if rows.iter().any(|r| !ids.insert(r.id)) {
        return Err("row IDs must be unique");
    }
    Ok(())
}

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn stratified_folds(rows: &[Row], k: usize, seed: u64) -> Result<Vec<usize>, &'static str> {
    validate(rows)?;
    if k < 2 || k > rows.len() {
        return Err("k must be between two and the row count");
    }
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| (rows[i].y, mix64(rows[i].id ^ seed)));
    let mut seen = [0usize; 2];
    let mut folds = vec![usize::MAX; rows.len()];
    for i in order {
        folds[i] = seen[rows[i].y as usize] % k;
        seen[rows[i].y as usize] += 1;
    }
    Ok(folds)
}

#[derive(Clone, Copy, Debug)]
struct Scaler {
    mean: [f64; 2],
    scale: [f64; 2],
}

impl Scaler {
    fn fit(rows: &[Row]) -> Result<Self, &'static str> {
        validate(rows)?;
        let mut mean = [0.0; 2];
        for r in rows {
            mean[0] += r.x[0];
            mean[1] += r.x[1];
        }
        mean[0] /= rows.len() as f64;
        mean[1] /= rows.len() as f64;
        let mut variance = [0.0; 2];
        for r in rows {
            variance[0] += (r.x[0] - mean[0]).powi(2);
            variance[1] += (r.x[1] - mean[1]).powi(2);
        }
        let mut scale = [
            (variance[0] / rows.len() as f64).sqrt(),
            (variance[1] / rows.len() as f64).sqrt(),
        ];
        for s in &mut scale {
            if *s == 0.0 {
                *s = 1.0;
            }
        }
        if mean.iter().chain(&scale).any(|x| !x.is_finite()) {
            return Err("scaling overflow; rescale the raw features");
        }
        Ok(Self { mean, scale })
    }
    fn apply(self, x: [f64; 2]) -> [f64; 2] {
        [
            (x[0] - self.mean[0]) / self.scale[0],
            (x[1] - self.mean[1]) / self.scale[1],
        ]
    }
}

#[derive(Clone, Copy, Debug)]
struct Config {
    rate: f64,
    l2: f64,
    epochs: usize,
}

#[derive(Clone, Copy, Debug)]
struct Logistic {
    w: [f64; 2],
    b: f64,
    scaler: Scaler,
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

impl Logistic {
    fn fit(rows: &[Row], config: Config) -> Result<Self, &'static str> {
        validate(rows)?;
        if config.epochs == 0
            || !config.rate.is_finite()
            || config.rate <= 0.0
            || !config.l2.is_finite()
            || config.l2 < 0.0
        {
            return Err("training settings must be finite and in range");
        }
        let scaler = Scaler::fit(rows)?;
        let mut model = Self {
            w: [0.0; 2],
            b: 0.0,
            scaler,
        };
        for _ in 0..config.epochs {
            let mut dw = [0.0; 2];
            let mut db = 0.0;
            for r in rows {
                let x = scaler.apply(r.x);
                let error = sigmoid(model.w[0] * x[0] + model.w[1] * x[1] + model.b) - r.y as f64;
                dw[0] += error * x[0];
                dw[1] += error * x[1];
                db += error;
            }
            let n = rows.len() as f64;
            model.w[0] -= config.rate * (dw[0] / n + config.l2 * model.w[0]);
            model.w[1] -= config.rate * (dw[1] / n + config.l2 * model.w[1]);
            model.b -= config.rate * db / n;
        }
        if model.w.iter().any(|x| !x.is_finite()) || !model.b.is_finite() {
            return Err("training diverged; reduce the learning rate");
        }
        Ok(model)
    }
    fn probability(self, x: [f64; 2]) -> f64 {
        let x = self.scaler.apply(x);
        sigmoid(self.w[0] * x[0] + self.w[1] * x[1] + self.b)
    }
    fn predict(self, x: [f64; 2]) -> u8 {
        u8::from(self.probability(x) >= 0.5)
    }
}

fn accuracy(rows: &[Row], predict: impl Fn([f64; 2]) -> u8) -> f64 {
    rows.iter().filter(|r| predict(r.x) == r.y).count() as f64 / rows.len() as f64
}

fn majority_label(rows: &[Row]) -> u8 {
    u8::from(rows.iter().filter(|r| r.y == 1).count() * 2 >= rows.len())
}

fn cv_accuracy(
    rows: &[Row],
    folds: &[usize],
    k: usize,
    config: Config,
) -> Result<f64, &'static str> {
    validate(rows)?;
    if k < 2 || k > rows.len() {
        return Err("k must be between two and the row count");
    }
    if folds.len() != rows.len() || folds.iter().any(|&f| f >= k) {
        return Err("every row needs one valid fold");
    }
    let mut correct = 0usize;
    let mut total = 0usize;
    for held_out in 0..k {
        let train: Vec<_> = rows
            .iter()
            .zip(folds)
            .filter(|(_, &f)| f != held_out)
            .map(|(&r, _)| r)
            .collect();
        let valid: Vec<_> = rows
            .iter()
            .zip(folds)
            .filter(|(_, &f)| f == held_out)
            .map(|(&r, _)| r)
            .collect();
        if train.len() < 2 || valid.is_empty() {
            return Err("each fold needs training and validation rows");
        }
        let model = Logistic::fit(&train, config)?;
        correct += valid.iter().filter(|r| model.predict(r.x) == r.y).count();
        total += valid.len();
    }
    Ok(correct as f64 / total as f64)
}

fn search(
    rows: &[Row],
    folds: &[usize],
    k: usize,
    candidates: &[Config],
) -> Result<(Config, Vec<f64>), &'static str> {
    if candidates.is_empty() {
        return Err("search requires at least one candidate");
    }
    let mut scores = Vec::with_capacity(candidates.len());
    for &candidate in candidates {
        scores.push(cv_accuracy(rows, folds, k, candidate)?);
    }
    let best = scores
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
        .map(|(i, _)| candidates[i])
        .unwrap();
    Ok((best, scores))
}

const DEV: [Row; 24] = [
    Row {
        id: 1,
        x: [-2.0, -1.0],
        y: 0,
    },
    Row {
        id: 2,
        x: [-1.8, -0.6],
        y: 0,
    },
    Row {
        id: 3,
        x: [-1.5, -1.4],
        y: 0,
    },
    Row {
        id: 4,
        x: [-1.2, -0.4],
        y: 0,
    },
    Row {
        id: 5,
        x: [-1.0, -1.2],
        y: 0,
    },
    Row {
        id: 6,
        x: [-0.8, -0.2],
        y: 0,
    },
    Row {
        id: 7,
        x: [-0.6, -0.9],
        y: 0,
    },
    Row {
        id: 8,
        x: [-0.4, -0.3],
        y: 0,
    },
    Row {
        id: 9,
        x: [-0.2, -0.8],
        y: 0,
    },
    Row {
        id: 10,
        x: [0.1, -0.7],
        y: 0,
    },
    Row {
        id: 11,
        x: [0.2, -0.3],
        y: 0,
    },
    Row {
        id: 12,
        x: [0.5, -0.9],
        y: 0,
    },
    Row {
        id: 13,
        x: [-0.4, 1.1],
        y: 1,
    },
    Row {
        id: 14,
        x: [-0.1, 0.5],
        y: 1,
    },
    Row {
        id: 15,
        x: [0.2, 0.8],
        y: 1,
    },
    Row {
        id: 16,
        x: [0.4, 0.3],
        y: 1,
    },
    Row {
        id: 17,
        x: [0.6, 1.3],
        y: 1,
    },
    Row {
        id: 18,
        x: [0.8, 0.5],
        y: 1,
    },
    Row {
        id: 19,
        x: [1.0, 1.0],
        y: 1,
    },
    Row {
        id: 20,
        x: [1.2, 0.4],
        y: 1,
    },
    Row {
        id: 21,
        x: [1.4, 1.4],
        y: 1,
    },
    Row {
        id: 22,
        x: [1.6, 0.7],
        y: 1,
    },
    Row {
        id: 23,
        x: [1.8, 1.2],
        y: 1,
    },
    Row {
        id: 24,
        x: [2.0, 0.6],
        y: 1,
    },
];

const TEST: [Row; 8] = [
    Row {
        id: 101,
        x: [-1.7, -0.8],
        y: 0,
    },
    Row {
        id: 102,
        x: [-0.9, -0.5],
        y: 0,
    },
    Row {
        id: 103,
        x: [-0.1, -0.6],
        y: 0,
    },
    Row {
        id: 104,
        x: [0.4, -0.5],
        y: 0,
    },
    Row {
        id: 105,
        x: [-0.2, 0.8],
        y: 1,
    },
    Row {
        id: 106,
        x: [0.5, 0.7],
        y: 1,
    },
    Row {
        id: 107,
        x: [1.1, 0.6],
        y: 1,
    },
    Row {
        id: 108,
        x: [1.7, 1.0],
        y: 1,
    },
];

const CANDIDATES: [Config; 4] = [
    Config {
        rate: 0.05,
        l2: 0.0,
        epochs: 80,
    },
    Config {
        rate: 0.1,
        l2: 0.0,
        epochs: 120,
    },
    Config {
        rate: 0.1,
        l2: 0.01,
        epochs: 120,
    },
    Config {
        rate: 0.2,
        l2: 0.1,
        epochs: 80,
    },
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let folds = stratified_folds(&DEV, 4, 20260908)?;
    let baseline = majority_label(&DEV);
    println!("frozen development fold IDs: {:?}", folds);
    println!(
        "majority baseline on untouched test: {:.1}%",
        100.0 * accuracy(&TEST, |_| baseline)
    );
    let (best, scores) = search(&DEV, &folds, 4, &CANDIDATES)?;
    for (i, score) in scores.iter().enumerate() {
        println!(
            "candidate {i} {:?}: CV {:.1}%",
            CANDIDATES[i],
            100.0 * score
        );
    }
    let final_model = Logistic::fit(&DEV, best)?;
    println!(
        "selected {:?}; untouched test accuracy: {:.1}%",
        best,
        100.0 * accuracy(&TEST, |x| final_model.predict(x))
    );
    let errors: Vec<_> = TEST
        .iter()
        .filter(|r| final_model.predict(r.x) != r.y)
        .map(|r| r.id)
        .collect();
    println!("misclassified test row IDs: {errors:?}");
    println!("Fixture results test the workflow; they are not a deployment estimate.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_ids_and_small_feature_units_are_supported() {
        let folds = stratified_folds(&DEV, 4, 7).unwrap();
        let mut reversed = DEV.to_vec();
        reversed.reverse();
        let reversed_folds = stratified_folds(&reversed, 4, 7).unwrap();
        assert!(folds.iter().eq(reversed_folds.iter().rev()));
        let mut duplicate = DEV;
        duplicate[1].id = duplicate[0].id;
        assert!(stratified_folds(&duplicate, 4, 7).is_err());
        assert!(cv_accuracy(&[], &[], 0, CANDIDATES[0]).is_err());
        let tiny: Vec<_> = DEV
            .iter()
            .map(|r| Row {
                x: [r.x[0] * 1e-20, r.x[1] * 1e-20],
                ..*r
            })
            .collect();
        let normal = Scaler::fit(&DEV).unwrap();
        let scaled = Scaler::fit(&tiny).unwrap();
        for (a, b) in normal.apply(DEV[0].x).iter().zip(scaled.apply(tiny[0].x)) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn folds_are_reproducible_complete_and_stratified() -> Result<(), &'static str> {
        let a = stratified_folds(&DEV, 4, 7)?;
        let b = stratified_folds(&DEV, 4, 7)?;
        assert_eq!(a, b);
        assert!(a.iter().all(|&f| f < 4));
        for fold in 0..4 {
            let zeros = DEV
                .iter()
                .zip(&a)
                .filter(|(r, f)| r.y == 0 && **f == fold)
                .count();
            let ones = DEV
                .iter()
                .zip(&a)
                .filter(|(r, f)| r.y == 1 && **f == fold)
                .count();
            assert_eq!((zeros, ones), (3, 3));
        }
        Ok(())
    }
    #[test]
    fn selected_model_beats_majority_on_untouched_test() -> Result<(), &'static str> {
        let folds = stratified_folds(&DEV, 4, 20260908)?;
        let (best, _) = search(&DEV, &folds, 4, &CANDIDATES)?;
        let model = Logistic::fit(&DEV, best)?;
        assert!(accuracy(&TEST, |x| model.predict(x)) > accuracy(&TEST, |_| majority_label(&DEV)));
        Ok(())
    }
    #[test]
    fn bad_evaluation_inputs_are_rejected() {
        assert!(stratified_folds(&DEV, 1, 0).is_err());
        assert!(cv_accuracy(&DEV, &[0], 4, CANDIDATES[0]).is_err());
    }
}
