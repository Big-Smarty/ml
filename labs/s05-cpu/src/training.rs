//! Supplied one-output dense model and scalar derivative oracle, adapted from projects/ch27.
#[derive(Clone, Debug)]
pub struct Model {
    pub weights: Vec<f32>,
    pub bias: f32,
}

#[derive(Debug, PartialEq)]
pub struct Gradient {
    pub weights: Vec<f64>,
    pub bias: f64,
}

pub fn fixture(batch: usize, in_features: usize) -> (Vec<f32>, Vec<f32>) {
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

pub fn validate_training_data(
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
    pub fn validate_inputs(&self, inputs: &[f32], in_features: usize) -> Result<(), String> {
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

    pub fn predict(&self, features: &[f32]) -> f32 {
        self.bias
            + self
                .weights
                .iter()
                .zip(features)
                .map(|(weight, feature)| weight * feature)
                .sum::<f32>()
    }

    pub fn predict_batch(&self, inputs: &[f32], in_features: usize) -> Result<Vec<f32>, String> {
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
}

pub fn loss_and_gradient_sum(
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

pub fn mean_loss_and_gradient(
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

pub fn loss_and_gradient(
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
