// Local threads communicate gradients and activations for three parallel layouts.
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

fn demo() -> Result<(), Box<dyn std::error::Error>> {
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
