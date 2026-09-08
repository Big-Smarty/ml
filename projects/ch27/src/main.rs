//! Deterministic sharded linear-regression training and inference with scoped threads.
use std::thread;

#[derive(Clone, Debug)]
struct Model {
    weights: Vec<f32>,
    bias: f32,
}

#[derive(Debug, PartialEq)]
struct Gradient {
    weights: Vec<f64>,
    bias: f64,
}

fn fixture(batch: usize, in_features: usize) -> (Vec<f32>, Vec<f32>) {
    let mut inputs = Vec::with_capacity(batch * in_features);
    let mut targets = Vec::with_capacity(batch);
    for row in 0..batch {
        let mut target = 0.25;
        for feature in 0..in_features {
            let value = (((row * 17 + feature * 13) % 29) as f32 - 14.0) / 14.0;
            inputs.push(value);
            target += value * (feature as f32 + 1.0) * 0.3;
        }
        targets.push(target);
    }
    (inputs, targets)
}

fn validate_training_data(
    inputs: &[f32],
    targets: &[f32],
    in_features: usize,
) -> Result<usize, String> {
    if in_features == 0
        || !inputs.len().is_multiple_of(in_features)
        || inputs.len() / in_features != targets.len()
        || targets.is_empty()
    {
        return Err("expected nonempty inputs=[batch,in_features] and targets=[batch]".into());
    }
    if inputs.iter().chain(targets).any(|value| !value.is_finite()) {
        return Err("data must be finite".into());
    }
    Ok(targets.len())
}

impl Model {
    fn validate_inputs(&self, inputs: &[f32], in_features: usize) -> Result<(), String> {
        if in_features == 0
            || self.weights.len() != in_features
            || !inputs.len().is_multiple_of(in_features)
        {
            return Err("inference shape mismatch".into());
        }
        if !self.bias.is_finite()
            || self
                .weights
                .iter()
                .chain(inputs)
                .any(|value| !value.is_finite())
        {
            return Err("model and input must be finite".into());
        }
        Ok(())
    }

    fn predict(&self, features: &[f32]) -> f32 {
        self.bias
            + self
                .weights
                .iter()
                .zip(features)
                .map(|(weight, feature)| weight * feature)
                .sum::<f32>()
    }

    fn predict_batch(&self, inputs: &[f32], in_features: usize) -> Result<Vec<f32>, String> {
        self.validate_inputs(inputs, in_features)?;
        let predictions: Vec<_> = inputs
            .chunks_exact(in_features)
            .map(|features| self.predict(features))
            .collect();
        if predictions.iter().any(|value| !value.is_finite()) {
            return Err("prediction overflow; rescale the inputs".into());
        }
        Ok(predictions)
    }

    fn predict_batch_parallel(
        &self,
        inputs: &[f32],
        in_features: usize,
        threads: usize,
    ) -> Result<Vec<f32>, String> {
        self.validate_inputs(inputs, in_features)?;
        if threads == 0 {
            return Err("inference shape or thread count is invalid".into());
        }
        let batch = inputs.len() / in_features;
        if batch == 0 {
            return Ok(Vec::new());
        }
        let shards = threads.min(batch);
        let rows_per_shard = batch.div_ceil(shards);
        let mut predictions = vec![0.; batch];
        thread::scope(|scope| {
            let mut remaining_predictions = &mut predictions[..];
            for shard in 0..shards {
                let start = shard * rows_per_shard;
                if start >= batch {
                    break;
                }
                let end = start.saturating_add(rows_per_shard).min(batch);
                let shard_len = end - start;
                let (shard_predictions, rest) = remaining_predictions.split_at_mut(shard_len);
                remaining_predictions = rest;
                let shard_inputs = &inputs[start * in_features..end * in_features];
                scope.spawn(move || {
                    for (features, prediction) in shard_inputs
                        .chunks_exact(in_features)
                        .zip(shard_predictions)
                    {
                        *prediction = self.predict(features);
                    }
                });
            }
        });
        if predictions.iter().any(|value| !value.is_finite()) {
            return Err("prediction overflow; rescale the inputs".into());
        }
        Ok(predictions)
    }
}

fn loss_and_gradient_sum(
    model: &Model,
    inputs: &[f32],
    targets: &[f32],
    in_features: usize,
) -> (f64, Gradient) {
    let mut loss_sum = 0.0;
    let mut gradient_sum = Gradient {
        weights: vec![0.0; in_features],
        bias: 0.0,
    };
    for (features, &target) in inputs.chunks_exact(in_features).zip(targets) {
        let error = model.predict(features) as f64 - target as f64;
        loss_sum += error * error;
        gradient_sum.bias += 2.0 * error;
        for (weight_gradient, &feature) in gradient_sum.weights.iter_mut().zip(features) {
            *weight_gradient += 2.0 * error * feature as f64;
        }
    }
    (loss_sum, gradient_sum)
}

fn mean_loss_and_gradient(
    loss_sum: f64,
    mut gradient_sum: Gradient,
    batch: usize,
) -> Result<(f64, Gradient), String> {
    let scale = 1.0 / batch as f64;
    gradient_sum.bias *= scale;
    for weight_gradient in &mut gradient_sum.weights {
        *weight_gradient *= scale;
    }
    let loss = loss_sum * scale;
    if !loss.is_finite()
        || !gradient_sum.bias.is_finite()
        || gradient_sum.weights.iter().any(|value| !value.is_finite())
    {
        return Err("gradient overflow; rescale inputs or model".into());
    }
    Ok((loss, gradient_sum))
}

fn loss_and_gradient(
    model: &Model,
    inputs: &[f32],
    targets: &[f32],
    in_features: usize,
) -> Result<(f64, Gradient), String> {
    let batch = validate_training_data(inputs, targets, in_features)?;
    model.validate_inputs(inputs, in_features)?;
    let (loss_sum, gradient_sum) = loss_and_gradient_sum(model, inputs, targets, in_features);
    mean_loss_and_gradient(loss_sum, gradient_sum, batch)
}

fn loss_and_gradient_parallel(
    model: &Model,
    inputs: &[f32],
    targets: &[f32],
    in_features: usize,
    threads: usize,
) -> Result<(f64, Gradient), String> {
    let batch = validate_training_data(inputs, targets, in_features)?;
    model.validate_inputs(inputs, in_features)?;
    if threads == 0 {
        return Err("model shape or thread count is invalid".into());
    }
    let shards = threads.min(batch);
    let rows_per_shard = batch.div_ceil(shards);
    let partials = thread::scope(|scope| {
        let mut handles = Vec::new();
        for shard in 0..shards {
            let start = shard * rows_per_shard;
            let end = start.saturating_add(rows_per_shard).min(batch);
            if start < end {
                let shard_inputs = &inputs[start * in_features..end * in_features];
                let shard_targets = &targets[start..end];
                handles.push(scope.spawn(move || {
                    loss_and_gradient_sum(model, shard_inputs, shard_targets, in_features)
                }));
            }
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("gradient worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut loss_sum = 0.0;
    let mut gradient_sum = Gradient {
        weights: vec![0.0; in_features],
        bias: 0.0,
    };
    for (shard_loss_sum, shard_gradient_sum) in partials {
        loss_sum += shard_loss_sum;
        gradient_sum.bias += shard_gradient_sum.bias;
        for (total, shard) in gradient_sum
            .weights
            .iter_mut()
            .zip(shard_gradient_sum.weights)
        {
            *total += shard;
        }
    }
    mean_loss_and_gradient(loss_sum, gradient_sum, batch)
}

fn train_step(
    model: &mut Model,
    inputs: &[f32],
    targets: &[f32],
    in_features: usize,
    threads: usize,
    learning_rate: f32,
) -> Result<f64, String> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("learning rate must be finite and positive".into());
    }
    let (loss, gradient) =
        loss_and_gradient_parallel(model, inputs, targets, in_features, threads)?;
    if model
        .weights
        .iter()
        .zip(&gradient.weights)
        .any(|(weight, weight_gradient)| {
            !(weight - learning_rate * *weight_gradient as f32).is_finite()
        })
        || !(model.bias - learning_rate * gradient.bias as f32).is_finite()
    {
        return Err("update overflow; reduce the learning rate".into());
    }
    for (weight, weight_gradient) in model.weights.iter_mut().zip(gradient.weights) {
        *weight -= learning_rate * weight_gradient as f32;
    }
    model.bias -= learning_rate * gradient.bias as f32;
    Ok(loss)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let in_features = 4;
    let (inputs, targets) = fixture(64, in_features);
    let mut model = Model {
        weights: vec![0.; in_features],
        bias: 0.,
    };
    let scalar_before = loss_and_gradient(&model, &inputs, &targets, in_features)?.0;
    let before = loss_and_gradient_parallel(&model, &inputs, &targets, in_features, 4)?.0;
    if (before - scalar_before).abs() > 1e-6 + 1e-5 * scalar_before.abs() {
        return Err("parallel loss disagrees with scalar loss".into());
    }
    for _ in 0..40 {
        train_step(&mut model, &inputs, &targets, in_features, 4, 0.2)?;
    }
    let after = loss_and_gradient_parallel(&model, &inputs, &targets, in_features, 4)?.0;
    let predictions = model.predict_batch_parallel(&inputs[..8 * in_features], in_features, 4)?;
    let scalar_predictions = model.predict_batch(&inputs[..8 * in_features], in_features)?;
    if predictions != scalar_predictions {
        return Err("parallel inference disagrees with scalar inference".into());
    }
    println!("four-shard training MSE: {before:.6} -> {after:.6}");
    println!("model weights={:?} bias={:.4}", model.weights, model.bias);
    println!("parallel inference first 3={:?}", &predictions[..3]);
    println!("fixed shard boundaries and reduction order are reproducible for this build; other thread counts may change low bits");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-6 + 1e-5 * b.abs()
    }

    #[test]
    fn loss_gradient_and_update_match_nonzero_hand_example() {
        let mut model = Model {
            weights: vec![0.5, -1.0],
            bias: 0.25,
        };
        let inputs = [1.0, 2.0, 3.0, 1.0];
        let targets = [-1.0, 1.0];
        let (loss, gradient) = loss_and_gradient_parallel(&model, &inputs, &targets, 2, 8).unwrap();
        assert!(close(loss, 0.0625));
        assert!(close(gradient.weights[0], -1.0));
        assert!(close(gradient.weights[1], -0.75));
        assert!(close(gradient.bias, -0.5));
        train_step(&mut model, &inputs, &targets, 2, 8, 0.1).unwrap();
        assert!(close(model.weights[0] as f64, 0.6));
        assert!(close(model.weights[1] as f64, -0.925));
        assert!(close(model.bias as f64, 0.3));
    }

    #[test]
    fn invalid_arithmetic_is_rejected_before_committing_update() {
        let mut model = Model {
            weights: vec![0.0],
            bias: 0.0,
        };
        assert!(model.predict_batch(&[f32::NAN], 1).is_err());
        assert!(model.predict_batch_parallel(&[f32::NAN], 1, 2).is_err());
        assert!(model.predict_batch_parallel(&[1.0], 1, 0).is_err());
        assert!(loss_and_gradient_parallel(&model, &[1.0], &[f32::NAN], 1, 1).is_err());
        assert!(loss_and_gradient_parallel(&model, &[1.0], &[2.0], 1, 0).is_err());
        assert!(train_step(&mut model, &[1.0], &[2.0], 1, 2, f32::MAX).is_err());
        assert_eq!(model.weights, vec![0.0]);
        assert_eq!(model.bias, 0.0);
        model.weights[0] = f32::MAX;
        assert!(loss_and_gradient_parallel(&model, &[2.0], &[0.0], 1, 1).is_err());
    }

    #[test]
    fn parallel_predictions_and_gradients_agree_with_scalar() {
        let in_features = 3;
        let (inputs, targets) = fixture(17, in_features);
        let model = Model {
            weights: vec![0.2, -0.1, 0.4],
            bias: 0.3,
        };
        let scalar_predictions = model.predict_batch(&inputs, in_features).unwrap();
        let parallel_predictions = model
            .predict_batch_parallel(&inputs, in_features, 4)
            .unwrap();
        assert_eq!(scalar_predictions, parallel_predictions);
        let scalar = loss_and_gradient(&model, &inputs, &targets, in_features).unwrap();
        let parallel =
            loss_and_gradient_parallel(&model, &inputs, &targets, in_features, 4).unwrap();
        assert!(close(parallel.0, scalar.0));
        for (parallel_weight, scalar_weight) in parallel.1.weights.iter().zip(scalar.1.weights) {
            assert!(close(*parallel_weight, scalar_weight));
        }
        assert!(close(parallel.1.bias, scalar.1.bias));
    }

    #[test]
    fn parallel_training_reduces_loss_with_unequal_shards() {
        let in_features = 4;
        let (inputs, targets) = fixture(31, in_features);
        let mut model = Model {
            weights: vec![0.; in_features],
            bias: 0.,
        };
        let before = loss_and_gradient_parallel(&model, &inputs, &targets, in_features, 3)
            .unwrap()
            .0;
        for _ in 0..60 {
            train_step(&mut model, &inputs, &targets, in_features, 3, 0.2).unwrap();
        }
        assert!(
            loss_and_gradient_parallel(&model, &inputs, &targets, in_features, 3)
                .unwrap()
                .0
                < before * 1e-3
        );
    }

    #[test]
    fn empty_prediction_batch_is_valid_but_empty_training_is_not() {
        let model = Model {
            weights: vec![1.],
            bias: 0.,
        };
        assert!(model.predict_batch_parallel(&[], 1, 2).unwrap().is_empty());
        assert!(loss_and_gradient_parallel(&model, &[], &[], 1, 2).is_err());
    }

    #[test]
    fn prediction_handles_every_short_uneven_partition() {
        let model = Model {
            weights: vec![0.5, -0.25],
            bias: 0.1,
        };
        for batch in 1..20 {
            let (inputs, _) = fixture(batch, 2);
            let scalar = model.predict_batch(&inputs, 2).unwrap();
            for threads in 1..25 {
                assert_eq!(
                    model.predict_batch_parallel(&inputs, 2, threads).unwrap(),
                    scalar
                );
            }
        }
    }
}
