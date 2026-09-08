//! Local threads communicate gradients and activations for three parallel layouts.
use std::sync::mpsc;
use std::thread;

const DATA: [Example; 6] = [
    Example {
        features: [-1.0, 0.0, 1.0, 0.5],
        target: -2.0,
    },
    Example {
        features: [0.0, 1.0, -1.0, 1.0],
        target: 2.5,
    },
    Example {
        features: [1.0, 0.5, 0.0, -1.0],
        target: 1.0,
    },
    Example {
        features: [2.0, -1.0, 0.5, 0.0],
        target: 5.0,
    },
    Example {
        features: [-0.5, 2.0, 1.0, -1.0],
        target: -1.5,
    },
    Example {
        features: [1.5, 1.0, -0.5, 2.0],
        target: 4.0,
    },
];

#[derive(Clone, Copy, Debug)]
struct Example {
    features: [f64; 4],
    target: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Model {
    weights: [f64; 4],
    bias: f64,
}

impl Model {
    fn predict(&self, features: &[f64; 4]) -> f64 {
        self.weights
            .iter()
            .zip(features)
            .map(|(weight, feature)| weight * feature)
            .sum::<f64>()
            + self.bias
    }
}

#[derive(Clone, Copy, Debug)]
struct Gradient {
    weights: [f64; 4],
    bias: f64,
    count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TrainingState {
    model: Model,
    step: usize,
    cursor: usize,
}

fn validate_data(data: &[Example]) -> Result<(), &'static str> {
    if data.is_empty() {
        return Err("training data must not be empty");
    }
    if data.iter().any(|example| {
        !example.target.is_finite() || example.features.iter().any(|value| !value.is_finite())
    }) {
        return Err("training data must be finite");
    }
    Ok(())
}

fn gradient_sum(model: &Model, batch: &[Example]) -> Gradient {
    let mut result = Gradient {
        weights: [0.0; 4],
        bias: 0.0,
        count: batch.len(),
    };
    for example in batch {
        let residual = model.predict(&example.features) - example.target;
        for (weight_sum, feature) in result.weights.iter_mut().zip(example.features) {
            *weight_sum += 2.0 * residual * feature;
        }
        result.bias += 2.0 * residual;
    }
    result
}

fn serial_step(model: Model, data: &[Example], learning_rate: f64) -> Result<Model, &'static str> {
    validate_data(data)?;
    apply_gradient_sums(model, &[gradient_sum(&model, data)], learning_rate)
}

fn data_parallel_step(
    model: Model,
    data: &[Example],
    learning_rate: f64,
) -> Result<Model, &'static str> {
    validate_data(data)?;
    let midpoint = data.len().div_ceil(2);
    let (tx, rx) = mpsc::channel();
    thread::scope(|scope| -> Result<(), &'static str> {
        let mut handles = Vec::new();
        for (worker, batch) in [&data[..midpoint], &data[midpoint..]]
            .into_iter()
            .enumerate()
        {
            let sender = tx.clone();
            handles.push(scope.spawn(move || sender.send((worker, gradient_sum(&model, batch)))));
        }
        drop(tx);
        for handle in handles {
            handle
                .join()
                .map_err(|_| "data worker panicked")?
                .map_err(|_| "gradient receiver closed")?;
        }
        Ok(())
    })?;
    let mut parts: Vec<_> = rx.into_iter().collect();
    parts.sort_by_key(|(worker, _)| *worker);
    let gradients: Vec<_> = parts.into_iter().map(|(_, gradient)| gradient).collect();
    apply_gradient_sums(model, &gradients, learning_rate)
}

fn aggregate_gradient_sums(parts: &[Gradient]) -> Result<Gradient, &'static str> {
    let count = parts.iter().map(|part| part.count).sum();
    if count == 0 {
        return Err("at least one worker example is required");
    }
    let mut total = Gradient {
        weights: [0.0; 4],
        bias: 0.0,
        count,
    };
    for part in parts {
        for (sum, value) in total.weights.iter_mut().zip(part.weights) {
            *sum += value;
        }
        total.bias += part.bias;
    }
    Ok(total)
}

fn apply_gradient_sums(
    model: Model,
    parts: &[Gradient],
    learning_rate: f64,
) -> Result<Model, &'static str> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("learning rate must be finite and positive");
    }
    let total = aggregate_gradient_sums(parts)?;
    let scale = learning_rate / total.count as f64;
    let mut next = model;
    for (weight, gradient_sum) in next.weights.iter_mut().zip(total.weights) {
        *weight -= scale * gradient_sum;
    }
    next.bias -= scale * total.bias;
    if next.weights.iter().any(|value| !value.is_finite()) || !next.bias.is_finite() {
        return Err("parameter update became nonfinite");
    }
    Ok(next)
}

fn tensor_parallel_step(
    model: Model,
    data: &[Example],
    learning_rate: f64,
) -> Result<Model, &'static str> {
    validate_data(data)?;
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("learning rate must be finite and positive");
    }
    let residuals = data
        .iter()
        .map(|example| Ok(tensor_parallel_predict(&model, &example.features)? - example.target))
        .collect::<Result<Vec<_>, &'static str>>()?;
    let (tx, rx) = mpsc::channel();
    thread::scope(|scope| -> Result<(), &'static str> {
        let mut handles = Vec::new();
        for shard in 0..2 {
            let sender = tx.clone();
            let residuals = &residuals;
            handles.push(scope.spawn(move || {
                let start = shard * 2;
                let mut gradient = [0.0; 2];
                for (example, residual) in data.iter().zip(residuals) {
                    for (local, value) in gradient.iter_mut().enumerate() {
                        *value += 2.0 * residual * example.features[start + local];
                    }
                }
                sender.send((shard, gradient))
            }));
        }
        drop(tx);
        for handle in handles {
            handle
                .join()
                .map_err(|_| "tensor gradient worker panicked")?
                .map_err(|_| "tensor gradient receiver closed")?;
        }
        Ok(())
    })?;
    let mut shards: Vec<_> = rx.into_iter().collect();
    shards.sort_by_key(|(id, _)| *id);
    let scale = learning_rate / data.len() as f64;
    let mut next = model;
    for (shard, gradient) in shards {
        for (local, value) in gradient.into_iter().enumerate() {
            next.weights[shard * 2 + local] -= scale * value;
        }
    }
    next.bias -= scale * residuals.iter().map(|residual| 2.0 * residual).sum::<f64>();
    if next.weights.iter().any(|value| !value.is_finite()) || !next.bias.is_finite() {
        return Err("tensor-parallel update became nonfinite");
    }
    Ok(next)
}

fn tensor_parallel_predict(model: &Model, features: &[f64; 4]) -> Result<f64, &'static str> {
    let model = *model;
    let features = *features;
    let (tx, rx) = mpsc::channel();
    let handles: Vec<_> = (0..2)
        .map(|shard| {
            let sender = tx.clone();
            thread::spawn(move || {
                let start = shard * 2;
                let partial: f64 = (start..start + 2)
                    .map(|index| model.weights[index] * features[index])
                    .sum();
                sender.send((shard, partial))
            })
        })
        .collect();
    drop(tx);
    for handle in handles {
        handle
            .join()
            .map_err(|_| "tensor worker panicked")?
            .map_err(|_| "tensor receiver closed")?;
    }
    let mut partials: Vec<_> = rx.into_iter().collect();
    partials.sort_by_key(|(shard, _)| *shard);
    Ok(partials.iter().map(|(_, value)| value).sum::<f64>() + model.bias)
}

fn pipeline_losses(model: Model, data: &[Example]) -> Result<Vec<f64>, &'static str> {
    validate_data(data)?;
    let (activation_tx, activation_rx) = mpsc::channel();
    let (loss_tx, loss_rx) = mpsc::channel();
    thread::scope(|scope| -> Result<(), &'static str> {
        let first = scope.spawn(move || -> Result<(), &'static str> {
            for (id, example) in data.iter().copied().enumerate() {
                activation_tx
                    .send((id, model.predict(&example.features), example.target))
                    .map_err(|_| "pipeline stage two closed")?;
            }
            Ok(())
        });
        let second = scope.spawn(move || -> Result<(), &'static str> {
            for (id, prediction, target) in activation_rx {
                loss_tx
                    .send((id, (prediction - target).powi(2)))
                    .map_err(|_| "pipeline collector closed")?;
            }
            Ok(())
        });
        first.join().map_err(|_| "pipeline stage one panicked")??;
        second.join().map_err(|_| "pipeline stage two panicked")??;
        Ok(())
    })?;
    let mut losses: Vec<_> = loss_rx.into_iter().collect();
    losses.sort_by_key(|(id, _)| *id);
    Ok(losses.into_iter().map(|(_, loss)| loss).collect())
}

#[derive(Clone, Copy, Debug)]
struct TwoLayer {
    hidden_weights: [f64; 2],
    hidden_biases: [f64; 2],
    output_weights: [f64; 2],
    output_bias: f64,
}

const PIPELINE_DATA: [(f64, f64); 4] = [(-1.0, -0.5), (-0.3, 0.2), (0.4, 0.9), (1.0, 1.5)];

fn apply_two_layer(
    mut model: TwoLayer,
    hidden_weight_sums: [f64; 2],
    hidden_bias_sums: [f64; 2],
    output_weight_sums: [f64; 2],
    output_bias_sum: f64,
    count: usize,
    learning_rate: f64,
) -> Result<TwoLayer, &'static str> {
    let scale = learning_rate / count as f64;
    for hidden in 0..2 {
        model.hidden_weights[hidden] -= scale * hidden_weight_sums[hidden];
        model.hidden_biases[hidden] -= scale * hidden_bias_sums[hidden];
        model.output_weights[hidden] -= scale * output_weight_sums[hidden];
    }
    model.output_bias -= scale * output_bias_sum;
    if model
        .hidden_weights
        .into_iter()
        .chain(model.hidden_biases)
        .chain(model.output_weights)
        .chain([model.output_bias])
        .any(|value| !value.is_finite())
    {
        return Err("pipeline update became nonfinite");
    }
    Ok(model)
}

fn validate_pipeline_data(data: &[(f64, f64)]) -> Result<(), &'static str> {
    if data.is_empty() {
        return Err("pipeline data must not be empty");
    }
    if data
        .iter()
        .any(|(input, target)| !input.is_finite() || !target.is_finite())
    {
        return Err("pipeline data must be finite");
    }
    Ok(())
}

fn serial_pipeline_step(
    model: TwoLayer,
    data: &[(f64, f64)],
    learning_rate: f64,
) -> Result<TwoLayer, &'static str> {
    validate_pipeline_data(data)?;
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("learning rate must be finite and positive");
    }
    let mut hidden_weight_sums = [0.0; 2];
    let mut hidden_bias_sums = [0.0; 2];
    let mut output_weight_sums = [0.0; 2];
    let mut output_bias_sum = 0.0;
    for &(input, target) in data {
        let hidden_activations = [
            (model.hidden_weights[0] * input + model.hidden_biases[0]).tanh(),
            (model.hidden_weights[1] * input + model.hidden_biases[1]).tanh(),
        ];
        let residual = model.output_weights[0] * hidden_activations[0]
            + model.output_weights[1] * hidden_activations[1]
            + model.output_bias
            - target;
        for hidden in 0..2 {
            output_weight_sums[hidden] += 2.0 * residual * hidden_activations[hidden];
            let hidden_pre_activation_gradient = 2.0
                * residual
                * model.output_weights[hidden]
                * (1.0 - hidden_activations[hidden].powi(2));
            hidden_weight_sums[hidden] += hidden_pre_activation_gradient * input;
            hidden_bias_sums[hidden] += hidden_pre_activation_gradient;
        }
        output_bias_sum += 2.0 * residual;
    }
    apply_two_layer(
        model,
        hidden_weight_sums,
        hidden_bias_sums,
        output_weight_sums,
        output_bias_sum,
        data.len(),
        learning_rate,
    )
}

fn pipeline_training_step(
    model: TwoLayer,
    data: &[(f64, f64)],
    learning_rate: f64,
) -> Result<TwoLayer, &'static str> {
    validate_pipeline_data(data)?;
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("learning rate must be finite and positive");
    }
    let (forward_tx, forward_rx) = mpsc::channel();
    let (backward_tx, backward_rx) = mpsc::channel::<(usize, f64, [f64; 2], [f64; 2])>();
    let (first_gradient_tx, first_gradient_rx) = mpsc::channel();
    let (second_gradient_tx, second_gradient_rx) = mpsc::channel();
    thread::scope(|scope| -> Result<(), &'static str> {
        let first = scope.spawn(move || -> Result<(), &'static str> {
            for (id, &(input, target)) in data.iter().enumerate() {
                let hidden_activations = [
                    (model.hidden_weights[0] * input + model.hidden_biases[0]).tanh(),
                    (model.hidden_weights[1] * input + model.hidden_biases[1]).tanh(),
                ];
                forward_tx
                    .send((id, input, target, hidden_activations))
                    .map_err(|_| "second stage closed")?;
            }
            drop(forward_tx);
            let mut hidden_weight_sums = [0.0; 2];
            let mut hidden_bias_sums = [0.0; 2];
            for (_id, input, hidden_activations, hidden_activation_gradients) in backward_rx {
                for hidden in 0..2 {
                    let hidden_pre_activation_gradient = hidden_activation_gradients[hidden]
                        * (1.0 - hidden_activations[hidden].powi(2));
                    hidden_weight_sums[hidden] += hidden_pre_activation_gradient * input;
                    hidden_bias_sums[hidden] += hidden_pre_activation_gradient;
                }
            }
            first_gradient_tx
                .send((hidden_weight_sums, hidden_bias_sums))
                .map_err(|_| "coordinator closed")
        });
        let second = scope.spawn(move || -> Result<(), &'static str> {
            let mut output_weight_sums = [0.0; 2];
            let mut output_bias_sum = 0.0;
            for (id, input, target, hidden_activations) in forward_rx {
                let residual = model.output_weights[0] * hidden_activations[0]
                    + model.output_weights[1] * hidden_activations[1]
                    + model.output_bias
                    - target;
                for hidden in 0..2 {
                    output_weight_sums[hidden] += 2.0 * residual * hidden_activations[hidden];
                }
                output_bias_sum += 2.0 * residual;
                backward_tx
                    .send((
                        id,
                        input,
                        hidden_activations,
                        [
                            2.0 * residual * model.output_weights[0],
                            2.0 * residual * model.output_weights[1],
                        ],
                    ))
                    .map_err(|_| "first stage closed")?;
            }
            second_gradient_tx
                .send((output_weight_sums, output_bias_sum))
                .map_err(|_| "coordinator closed")
        });
        first.join().map_err(|_| "first stage panicked")??;
        second.join().map_err(|_| "second stage panicked")??;
        Ok(())
    })?;
    let (hidden_weight_sums, hidden_bias_sums) = first_gradient_rx
        .recv()
        .map_err(|_| "missing first-stage gradients")?;
    let (output_weight_sums, output_bias_sum) = second_gradient_rx
        .recv()
        .map_err(|_| "missing second-stage gradients")?;
    apply_two_layer(
        model,
        hidden_weight_sums,
        hidden_bias_sums,
        output_weight_sums,
        output_bias_sum,
        data.len(),
        learning_rate,
    )
}

impl TrainingState {
    fn train_step(
        mut self,
        data: &[Example],
        batch_size: usize,
        learning_rate: f64,
    ) -> Result<Self, &'static str> {
        validate_data(data)?;
        if batch_size == 0 || batch_size > data.len() {
            return Err("batch size must be in 1..=data length");
        }
        if self.cursor >= data.len() {
            return Err("checkpoint cursor is outside the dataset");
        }
        let batch: Vec<_> = (0..batch_size)
            .map(|offset| {
                self.cursor
                    .checked_add(offset)
                    .map(|index| data[index % data.len()])
                    .ok_or("minibatch index overflow")
            })
            .collect::<Result<_, _>>()?;
        self.model = data_parallel_step(self.model, &batch, learning_rate)?;
        self.step = self.step.checked_add(1).ok_or("training step overflow")?;
        self.cursor = self
            .cursor
            .checked_add(batch_size)
            .ok_or("minibatch cursor overflow")?
            % data.len();
        Ok(self)
    }

    fn encode(self) -> String {
        format!(
            "CH50v1 {} {} {} {} {} {} {}",
            self.step,
            self.cursor,
            self.model.weights[0],
            self.model.weights[1],
            self.model.weights[2],
            self.model.weights[3],
            self.model.bias
        )
    }

    fn decode(text: &str) -> Result<Self, &'static str> {
        let fields: Vec<_> = text.split_whitespace().collect();
        if fields.len() != 8 || fields[0] != "CH50v1" {
            return Err("unsupported or malformed checkpoint");
        }
        let step = fields[1].parse().map_err(|_| "invalid checkpoint step")?;
        let cursor = fields[2].parse().map_err(|_| "invalid checkpoint cursor")?;
        let mut numbers: [f64; 5] = [0.0; 5];
        for (slot, field) in numbers.iter_mut().zip(&fields[3..]) {
            *slot = field.parse().map_err(|_| "invalid checkpoint parameter")?;
        }
        if numbers.iter().any(|v| !v.is_finite()) {
            return Err("checkpoint parameters must be finite");
        }
        Ok(Self {
            model: Model {
                weights: [numbers[0], numbers[1], numbers[2], numbers[3]],
                bias: numbers[4],
            },
            step,
            cursor,
        })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let initial = TrainingState {
        model: Model {
            weights: [0.0; 4],
            bias: 0.0,
        },
        step: 0,
        cursor: 0,
    };
    let after_one = initial.train_step(&DATA, 3, 0.05)?;
    let restored = TrainingState::decode(&after_one.encode())?;
    let resumed = restored.train_step(&DATA, 3, 0.05)?;
    let uninterrupted = initial
        .train_step(&DATA, 3, 0.05)?
        .train_step(&DATA, 3, 0.05)?;
    println!("data parallel: two workers sent private gradient sums");
    println!(
        "tensor parallel prediction: {:.6}",
        tensor_parallel_predict(&resumed.model, &DATA[0].features)?
    );
    let tensor_update = tensor_parallel_step(resumed.model, &DATA[..5], 0.02)?;
    let serial_update = serial_step(resumed.model, &DATA[..5], 0.02)?;
    println!(
        "tensor-parallel update matches serial within 1e-12: {}",
        tensor_update
            .weights
            .iter()
            .zip(serial_update.weights)
            .all(|(a, b)| (a - b).abs() < 1e-12)
            && (tensor_update.bias - serial_update.bias).abs() < 1e-12
    );
    let losses = pipeline_losses(resumed.model, &DATA)?;
    println!(
        "pipeline stages communicated {} activations; mean loss {:.6}",
        losses.len(),
        losses.iter().sum::<f64>() / losses.len() as f64
    );
    let two_layer = TwoLayer {
        hidden_weights: [0.2, -0.3],
        hidden_biases: [0.1, 0.0],
        output_weights: [0.4, -0.2],
        output_bias: 0.0,
    };
    let trained_pipeline = pipeline_training_step(two_layer, &PIPELINE_DATA, 0.05)?;
    let serial_pipeline = serial_pipeline_step(two_layer, &PIPELINE_DATA, 0.05)?;
    println!(
        "pipeline backward updated both stages and matches serial: {}",
        trained_pipeline.hidden_weights != two_layer.hidden_weights
            && trained_pipeline.output_weights != two_layer.output_weights
            && trained_pipeline
                .hidden_weights
                .into_iter()
                .chain(trained_pipeline.hidden_biases)
                .chain(trained_pipeline.output_weights)
                .chain([trained_pipeline.output_bias])
                .zip(
                    serial_pipeline
                        .hidden_weights
                        .into_iter()
                        .chain(serial_pipeline.hidden_biases)
                        .chain(serial_pipeline.output_weights)
                        .chain([serial_pipeline.output_bias]),
                )
                .all(|(a, b)| (a - b).abs() < 1e-12)
    );
    println!("recovery equivalence: {}", resumed == uninterrupted);
    println!("This is measured local CPU thread communication, not multi-GPU execution.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models_are_close(left: Model, right: Model) -> bool {
        left.weights
            .iter()
            .zip(right.weights)
            .all(|(left, right)| (left - right).abs() < 1e-12)
            && (left.bias - right.bias).abs() < 1e-12
    }

    fn two_layer_values(model: TwoLayer) -> impl Iterator<Item = f64> {
        model
            .hidden_weights
            .into_iter()
            .chain(model.hidden_biases)
            .chain(model.output_weights)
            .chain([model.output_bias])
    }

    fn pipeline_mean_squared_error(model: TwoLayer, data: &[(f64, f64)]) -> f64 {
        data.iter()
            .map(|&(input, target)| {
                let prediction = model.output_weights[0]
                    * (model.hidden_weights[0] * input + model.hidden_biases[0]).tanh()
                    + model.output_weights[1]
                        * (model.hidden_weights[1] * input + model.hidden_biases[1]).tanh()
                    + model.output_bias;
                (prediction - target).powi(2)
            })
            .sum::<f64>()
            / data.len() as f64
    }

    #[test]
    fn parallel_step_matches_serial_and_shards_match_dot() -> Result<(), &'static str> {
        let model = Model {
            weights: [0.2, -0.1, 0.3, 0.4],
            bias: -0.2,
        };
        for size in [1, 3, 5] {
            assert!(models_are_close(
                data_parallel_step(model, &DATA[..size], 0.03)?,
                serial_step(model, &DATA[..size], 0.03)?
            ));
        }
        assert!(models_are_close(
            tensor_parallel_step(model, &DATA[..5], 0.03)?,
            serial_step(model, &DATA[..5], 0.03)?
        ));
        assert!(
            (tensor_parallel_predict(&model, &DATA[2].features)?
                - model.predict(&DATA[2].features))
            .abs()
                < 1e-12
        );
        Ok(())
    }

    #[test]
    fn pipeline_and_recovery_preserve_results() -> Result<(), &'static str> {
        let initial = TrainingState {
            model: Model {
                weights: [0.0; 4],
                bias: 0.0,
            },
            step: 0,
            cursor: 0,
        };
        let direct: Vec<_> = DATA
            .iter()
            .map(|example| (initial.model.predict(&example.features) - example.target).powi(2))
            .collect();
        assert_eq!(pipeline_losses(initial.model, &DATA)?, direct);
        let first = initial.train_step(&DATA, 3, 0.05)?;
        assert_eq!(first.cursor, 3);
        assert_eq!(
            TrainingState {
                model: Model {
                    weights: [1.0, 2.0, 3.0, 4.0],
                    bias: 5.0,
                },
                step: 7,
                cursor: 2,
            }
            .encode(),
            "CH50v1 7 2 1 2 3 4 5"
        );
        let resumed = TrainingState::decode(&first.encode())?.train_step(&DATA, 3, 0.05)?;
        let continuous = first.train_step(&DATA, 3, 0.05)?;
        assert_eq!(resumed, continuous);
        for record in [
            "CH50v2 1 0 0 0 0 0 0",
            "CH50v1 1 0 0 0 0 0",
            "CH50v1 bad 0 0 0 0 0 0",
            "CH50v1 1 0 NaN 0 0 0 0",
            "CH50v1 1 0 0 0 0 0 inf",
        ] {
            assert!(TrainingState::decode(record).is_err());
        }
        let exhausted = TrainingState {
            step: usize::MAX,
            ..first
        };
        assert!(exhausted.train_step(&DATA, 3, 0.05).is_err());
        let bad_cursor = TrainingState::decode("CH50v1 1 99 0 0 0 0 0")?;
        assert!(bad_cursor.train_step(&DATA, 3, 0.05).is_err());
        Ok(())
    }

    #[test]
    fn pipeline_backward_matches_serial_update() -> Result<(), &'static str> {
        let model = TwoLayer {
            hidden_weights: [0.2, -0.3],
            hidden_biases: [0.1, 0.0],
            output_weights: [0.4, -0.2],
            output_bias: 0.0,
        };
        for data in [&PIPELINE_DATA[..3], &PIPELINE_DATA[..]] {
            let parallel = pipeline_training_step(model, data, 0.05)?;
            let serial = serial_pipeline_step(model, data, 0.05)?;
            for (parallel_value, serial_value) in
                two_layer_values(parallel).zip(two_layer_values(serial))
            {
                assert!((parallel_value - serial_value).abs() < 1e-12);
            }

            let mut plus = model;
            let mut minus = model;
            plus.hidden_weights[0] += 1e-5;
            minus.hidden_weights[0] -= 1e-5;
            let numerical = (pipeline_mean_squared_error(plus, data)
                - pipeline_mean_squared_error(minus, data))
                / 2e-5;
            let analytic = (model.hidden_weights[0] - parallel.hidden_weights[0]) / 0.05;
            assert!((analytic - numerical).abs() < 1e-6 + 1e-4 * numerical.abs());
        }

        assert!(pipeline_training_step(model, &[], 0.05).is_err());
        assert!(serial_pipeline_step(model, &[(f64::NAN, 0.0)], 0.05).is_err());
        assert!(pipeline_training_step(model, &PIPELINE_DATA, 0.0).is_err());
        assert!(serial_pipeline_step(model, &[(1.0, -f64::MAX)], 1.0).is_err());
        Ok(())
    }
}
