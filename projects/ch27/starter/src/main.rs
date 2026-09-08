#[cfg(test)]
use std::thread;

#[derive(Clone, Debug)]
struct Model {
    weights: Vec<f32>,
    bias: f32,
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

    fn loss(&self, inputs: &[f32], targets: &[f32], in_features: usize) -> Result<f64, String> {
        let batch = validate_training_data(inputs, targets, in_features)?;
        self.validate_inputs(inputs, in_features)?;
        let loss = inputs
            .chunks_exact(in_features)
            .zip(targets)
            .map(|(features, &target)| {
                let error = self.predict(features) as f64 - target as f64;
                error * error
            })
            .sum::<f64>()
            / batch as f64;
        if !loss.is_finite() {
            return Err("loss overflow; rescale inputs or model".into());
        }
        Ok(loss)
    }

    #[cfg(test)]
    fn loss_parallel(
        &self,
        inputs: &[f32],
        targets: &[f32],
        in_features: usize,
        threads: usize,
    ) -> Result<f64, String> {
        validate_training_data(inputs, targets, in_features)?;
        self.validate_inputs(inputs, in_features)?;
        if threads == 0 {
            return Err("model shape or thread count is invalid".into());
        }
        // TODO: return a sum from each worker, reduce in handle order, then divide once by batch size.
        thread::scope(|_| {});
        todo!("implement fixed-order shard reduction")
    }
}

fn main() -> Result<(), String> {
    let inputs = [1., 2., 2., 1.];
    let targets = [5., 4.];
    let model = Model {
        weights: vec![1., 2.],
        bias: 0.0,
    };
    println!(
        "prior scalar inference MSE={}",
        model.loss(&inputs, &targets, 2)?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parallel_reduction_matches_scalar() {
        let inputs = [1., 2., 2., 1., 3., -1.];
        let targets = [5., 4., 2.];
        let model = Model {
            weights: vec![1., 2.],
            bias: 0.0,
        };
        assert!(
            (model.loss_parallel(&inputs, &targets, 2, 2).unwrap()
                - model.loss(&inputs, &targets, 2).unwrap())
            .abs()
                < 1e-12
        );
    }
}
