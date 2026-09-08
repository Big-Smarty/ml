//! Reproducible cross-validation and bounded search on a bundled tabular fixture.

#[derive(Clone, Copy, Debug)]
struct Row {
    id: u64,
    features: [f64; 2],
    label: u8,
}

fn validate(rows: &[Row]) -> Result<(), &'static str> {
    if rows.len() < 2 {
        return Err("evaluation requires at least two rows");
    }
    if rows
        .iter()
        .any(|row| row.label > 1 || row.features.iter().any(|value| !value.is_finite()))
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
    order.sort_by_key(|&i| (rows[i].label, mix64(rows[i].id ^ seed)));
    let mut seen = [0usize; 2];
    let mut folds = vec![usize::MAX; rows.len()];
    for i in order {
        folds[i] = seen[rows[i].label as usize] % k;
        seen[rows[i].label as usize] += 1;
    }
    Ok(folds)
}

#[derive(Clone, Copy, Debug)]
struct Scaler {
    mean: [f64; 2],
    scale: [f64; 2],
}

impl Scaler {
    fn fit(train_data: &[Row]) -> Result<Self, &'static str> {
        validate(train_data)?;
        let mut mean = [0.0; 2];
        for row in train_data {
            mean[0] += row.features[0];
            mean[1] += row.features[1];
        }
        mean[0] /= train_data.len() as f64;
        mean[1] /= train_data.len() as f64;
        let mut variance = [0.0; 2];
        for row in train_data {
            variance[0] += (row.features[0] - mean[0]).powi(2);
            variance[1] += (row.features[1] - mean[1]).powi(2);
        }
        let mut scale = [
            (variance[0] / train_data.len() as f64).sqrt(),
            (variance[1] / train_data.len() as f64).sqrt(),
        ];
        for s in &mut scale {
            if *s == 0.0 {
                *s = 1.0;
            }
        }
        if mean.iter().chain(&scale).any(|value| !value.is_finite()) {
            return Err("scaling overflow; rescale the raw features");
        }
        Ok(Self { mean, scale })
    }
    fn transform(&self, features: [f64; 2]) -> [f64; 2] {
        [
            (features[0] - self.mean[0]) / self.scale[0],
            (features[1] - self.mean[1]) / self.scale[1],
        ]
    }
}

#[derive(Clone, Copy, Debug)]
struct Config {
    learning_rate: f64,
    l2: f64,
    epochs: usize,
}

#[derive(Clone, Copy, Debug)]
struct LogisticModel {
    weights: [f64; 2],
    bias: f64,
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

impl LogisticModel {
    fn fit(train_data: &[Row], config: Config) -> Result<Self, &'static str> {
        validate(train_data)?;
        if config.epochs == 0
            || !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
            || !config.l2.is_finite()
            || config.l2 < 0.0
        {
            return Err("training settings must be finite and in range");
        }
        let scaler = Scaler::fit(train_data)?;
        let mut model = Self {
            weights: [0.0; 2],
            bias: 0.0,
            scaler,
        };
        for _ in 0..config.epochs {
            let mut weight_gradient = [0.0; 2];
            let mut bias_gradient = 0.0;
            for row in train_data {
                let features = scaler.transform(row.features);
                let error = sigmoid(
                    model.weights[0] * features[0] + model.weights[1] * features[1] + model.bias,
                ) - row.label as f64;
                weight_gradient[0] += error * features[0];
                weight_gradient[1] += error * features[1];
                bias_gradient += error;
            }
            let n = train_data.len() as f64;
            model.weights[0] -=
                config.learning_rate * (weight_gradient[0] / n + config.l2 * model.weights[0]);
            model.weights[1] -=
                config.learning_rate * (weight_gradient[1] / n + config.l2 * model.weights[1]);
            model.bias -= config.learning_rate * bias_gradient / n;
        }
        if model.weights.iter().any(|value| !value.is_finite()) || !model.bias.is_finite() {
            return Err("training diverged; reduce the learning rate");
        }
        Ok(model)
    }
    fn probability(&self, features: [f64; 2]) -> f64 {
        let features = self.scaler.transform(features);
        sigmoid(self.weights[0] * features[0] + self.weights[1] * features[1] + self.bias)
    }
    fn predict(&self, features: [f64; 2]) -> u8 {
        u8::from(self.probability(features) >= 0.5)
    }
}

fn accuracy(rows: &[Row], predict: impl Fn([f64; 2]) -> u8) -> f64 {
    rows.iter()
        .filter(|row| predict(row.features) == row.label)
        .count() as f64
        / rows.len() as f64
}

fn majority_label(rows: &[Row]) -> u8 {
    u8::from(rows.iter().filter(|row| row.label == 1).count() * 2 >= rows.len())
}

fn pooled_cv_accuracy(
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
        let train_data: Vec<_> = rows
            .iter()
            .zip(folds)
            .filter(|(_, &f)| f != held_out)
            .map(|(&r, _)| r)
            .collect();
        let validation_data: Vec<_> = rows
            .iter()
            .zip(folds)
            .filter(|(_, &f)| f == held_out)
            .map(|(&r, _)| r)
            .collect();
        if train_data.len() < 2 || validation_data.is_empty() {
            return Err("each fold needs training and validation rows");
        }
        let model = LogisticModel::fit(&train_data, config)?;
        correct += validation_data
            .iter()
            .filter(|row| model.predict(row.features) == row.label)
            .count();
        total += validation_data.len();
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
        scores.push(pooled_cv_accuracy(rows, folds, k, candidate)?);
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
        features: [-2.0, -1.0],
        label: 0,
    },
    Row {
        id: 2,
        features: [-1.8, -0.6],
        label: 0,
    },
    Row {
        id: 3,
        features: [-1.5, -1.4],
        label: 0,
    },
    Row {
        id: 4,
        features: [-1.2, -0.4],
        label: 0,
    },
    Row {
        id: 5,
        features: [-1.0, -1.2],
        label: 0,
    },
    Row {
        id: 6,
        features: [-0.8, -0.2],
        label: 0,
    },
    Row {
        id: 7,
        features: [-0.6, -0.9],
        label: 0,
    },
    Row {
        id: 8,
        features: [-0.4, -0.3],
        label: 0,
    },
    Row {
        id: 9,
        features: [-0.2, -0.8],
        label: 0,
    },
    Row {
        id: 10,
        features: [0.1, -0.7],
        label: 0,
    },
    Row {
        id: 11,
        features: [0.2, -0.3],
        label: 0,
    },
    Row {
        id: 12,
        features: [0.5, -0.9],
        label: 0,
    },
    Row {
        id: 13,
        features: [-0.4, 1.1],
        label: 1,
    },
    Row {
        id: 14,
        features: [-0.1, 0.5],
        label: 1,
    },
    Row {
        id: 15,
        features: [0.2, 0.8],
        label: 1,
    },
    Row {
        id: 16,
        features: [0.4, 0.3],
        label: 1,
    },
    Row {
        id: 17,
        features: [0.6, 1.3],
        label: 1,
    },
    Row {
        id: 18,
        features: [0.8, 0.5],
        label: 1,
    },
    Row {
        id: 19,
        features: [1.0, 1.0],
        label: 1,
    },
    Row {
        id: 20,
        features: [1.2, 0.4],
        label: 1,
    },
    Row {
        id: 21,
        features: [1.4, 1.4],
        label: 1,
    },
    Row {
        id: 22,
        features: [1.6, 0.7],
        label: 1,
    },
    Row {
        id: 23,
        features: [1.8, 1.2],
        label: 1,
    },
    Row {
        id: 24,
        features: [2.0, 0.6],
        label: 1,
    },
];

const TEST: [Row; 8] = [
    Row {
        id: 101,
        features: [-1.7, -0.8],
        label: 0,
    },
    Row {
        id: 102,
        features: [-0.9, -0.5],
        label: 0,
    },
    Row {
        id: 103,
        features: [-0.1, -0.6],
        label: 0,
    },
    Row {
        id: 104,
        features: [0.4, -0.5],
        label: 0,
    },
    Row {
        id: 105,
        features: [-0.2, 0.8],
        label: 1,
    },
    Row {
        id: 106,
        features: [0.5, 0.7],
        label: 1,
    },
    Row {
        id: 107,
        features: [1.1, 0.6],
        label: 1,
    },
    Row {
        id: 108,
        features: [1.7, 1.0],
        label: 1,
    },
];

const CANDIDATES: [Config; 4] = [
    Config {
        learning_rate: 0.05,
        l2: 0.0,
        epochs: 80,
    },
    Config {
        learning_rate: 0.1,
        l2: 0.0,
        epochs: 120,
    },
    Config {
        learning_rate: 0.1,
        l2: 0.01,
        epochs: 120,
    },
    Config {
        learning_rate: 0.2,
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
    let final_model = LogisticModel::fit(&DEV, best)?;
    println!(
        "selected {:?}; untouched test accuracy: {:.1}%",
        best,
        100.0 * accuracy(&TEST, |features| final_model.predict(features))
    );
    let errors: Vec<_> = TEST
        .iter()
        .filter(|r| final_model.predict(r.features) != r.label)
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
        assert!(pooled_cv_accuracy(&[], &[], 0, CANDIDATES[0]).is_err());
        let tiny: Vec<_> = DEV
            .iter()
            .map(|r| Row {
                features: [r.features[0] * 1e-20, r.features[1] * 1e-20],
                ..*r
            })
            .collect();
        let normal = Scaler::fit(&DEV).unwrap();
        let scaled = Scaler::fit(&tiny).unwrap();
        for (a, b) in normal
            .transform(DEV[0].features)
            .iter()
            .zip(scaled.transform(tiny[0].features))
        {
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
                .filter(|(r, f)| r.label == 0 && **f == fold)
                .count();
            let ones = DEV
                .iter()
                .zip(&a)
                .filter(|(r, f)| r.label == 1 && **f == fold)
                .count();
            assert_eq!((zeros, ones), (3, 3));
        }
        Ok(())
    }
    #[test]
    fn selected_model_beats_majority_on_untouched_test() -> Result<(), &'static str> {
        let folds = stratified_folds(&DEV, 4, 20260908)?;
        let (best, _) = search(&DEV, &folds, 4, &CANDIDATES)?;
        let model = LogisticModel::fit(&DEV, best)?;
        assert!(
            accuracy(&TEST, |features| model.predict(features))
                > accuracy(&TEST, |_| majority_label(&DEV))
        );
        Ok(())
    }
    #[test]
    fn bad_evaluation_inputs_are_rejected() {
        assert!(stratified_folds(&DEV, 1, 0).is_err());
        assert!(pooled_cv_accuracy(&DEV, &[0], 4, CANDIDATES[0]).is_err());
    }
}
