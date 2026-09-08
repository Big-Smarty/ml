//! Local threads communicate gradients and activations for three parallel layouts.
use std::sync::mpsc;
use std::thread;

const DATA: [Example; 6] = [
    Example {
        x: [-1.0, 0.0, 1.0, 0.5],
        y: -2.0,
    },
    Example {
        x: [0.0, 1.0, -1.0, 1.0],
        y: 2.5,
    },
    Example {
        x: [1.0, 0.5, 0.0, -1.0],
        y: 1.0,
    },
    Example {
        x: [2.0, -1.0, 0.5, 0.0],
        y: 5.0,
    },
    Example {
        x: [-0.5, 2.0, 1.0, -1.0],
        y: -1.5,
    },
    Example {
        x: [1.5, 1.0, -0.5, 2.0],
        y: 4.0,
    },
];

#[derive(Clone, Copy, Debug)]
struct Example {
    x: [f64; 4],
    y: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Model {
    w: [f64; 4],
    b: f64,
}

#[derive(Clone, Copy, Debug)]
struct Gradient {
    worker: usize,
    dw: [f64; 4],
    db: f64,
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
    if data
        .iter()
        .any(|e| !e.y.is_finite() || e.x.iter().any(|v| !v.is_finite()))
    {
        return Err("training data must be finite");
    }
    Ok(())
}

fn dot(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn predict(model: Model, x: &[f64; 4]) -> f64 {
    dot(&model.w, x) + model.b
}

fn gradient_sum(worker: usize, model: Model, batch: &[Example]) -> Gradient {
    let mut result = Gradient {
        worker,
        dw: [0.0; 4],
        db: 0.0,
        count: batch.len(),
    };
    for example in batch {
        let residual = predict(model, &example.x) - example.y;
        for (dw, x) in result.dw.iter_mut().zip(example.x) {
            *dw += 2.0 * residual * x;
        }
        result.db += 2.0 * residual;
    }
    result
}

fn serial_step(model: Model, data: &[Example], rate: f64) -> Result<Model, &'static str> {
    validate_data(data)?;
    apply(model, &[gradient_sum(0, model, data)], rate)
}

fn data_parallel_step(model: Model, data: &[Example], rate: f64) -> Result<Model, &'static str> {
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
            handles.push(scope.spawn(move || sender.send(gradient_sum(worker, model, batch))));
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
    parts.sort_by_key(|part| part.worker);
    apply(model, &parts, rate)
}

fn apply(model: Model, parts: &[Gradient], rate: f64) -> Result<Model, &'static str> {
    if !rate.is_finite() || rate <= 0.0 {
        return Err("learning rate must be finite and positive");
    }
    let count: usize = parts.iter().map(|part| part.count).sum();
    if count == 0 {
        return Err("at least one worker example is required");
    }
    let mut total = [0.0; 4];
    for part in parts {
        for (sum, value) in total.iter_mut().zip(part.dw) {
            *sum += value;
        }
    }
    let scale = rate / count as f64;
    let mut next = model;
    for (weight, grad) in next.w.iter_mut().zip(total) {
        *weight -= scale * grad;
    }
    next.b -= scale * parts.iter().map(|part| part.db).sum::<f64>();
    if next.w.iter().any(|value| !value.is_finite()) || !next.b.is_finite() {
        return Err("parameter update became nonfinite");
    }
    Ok(next)
}

fn tensor_parallel_step(model: Model, data: &[Example], rate: f64) -> Result<Model, &'static str> {
    validate_data(data)?;
    if !rate.is_finite() || rate <= 0.0 {
        return Err("learning rate must be finite and positive");
    }
    let residuals = data
        .iter()
        .map(|example| Ok(tensor_parallel_predict(model, example.x)? - example.y))
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
                        *value += 2.0 * residual * example.x[start + local];
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
    let scale = rate / data.len() as f64;
    let mut next = model;
    for (shard, gradient) in shards {
        for (local, value) in gradient.into_iter().enumerate() {
            next.w[shard * 2 + local] -= scale * value;
        }
    }
    next.b -= scale * residuals.iter().map(|residual| 2.0 * residual).sum::<f64>();
    if next.w.iter().any(|value| !value.is_finite()) || !next.b.is_finite() {
        return Err("tensor-parallel update became nonfinite");
    }
    Ok(next)
}

fn tensor_parallel_predict(model: Model, x: [f64; 4]) -> Result<f64, &'static str> {
    let (tx, rx) = mpsc::channel();
    let handles: Vec<_> = (0..2)
        .map(|shard| {
            let sender = tx.clone();
            thread::spawn(move || {
                let start = shard * 2;
                let partial: f64 = (start..start + 2).map(|i| model.w[i] * x[i]).sum();
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
    Ok(partials.iter().map(|(_, value)| value).sum::<f64>() + model.b)
}

fn pipeline_losses(model: Model, data: &[Example]) -> Result<Vec<f64>, &'static str> {
    validate_data(data)?;
    let (activation_tx, activation_rx) = mpsc::channel();
    let (loss_tx, loss_rx) = mpsc::channel();
    thread::scope(|scope| -> Result<(), &'static str> {
        let first = scope.spawn(move || -> Result<(), &'static str> {
            for (id, example) in data.iter().copied().enumerate() {
                activation_tx
                    .send((id, predict(model, &example.x), example.y))
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
    w1: [f64; 2],
    b1: [f64; 2],
    w2: [f64; 2],
    b2: f64,
}

const PIPELINE_DATA: [(f64, f64); 4] = [(-1.0, -0.5), (-0.3, 0.2), (0.4, 0.9), (1.0, 1.5)];

fn apply_two_layer(
    mut model: TwoLayer,
    dw1: [f64; 2],
    db1: [f64; 2],
    dw2: [f64; 2],
    db2: f64,
    rate: f64,
) -> TwoLayer {
    let scale = rate / PIPELINE_DATA.len() as f64;
    for j in 0..2 {
        model.w1[j] -= scale * dw1[j];
        model.b1[j] -= scale * db1[j];
        model.w2[j] -= scale * dw2[j];
    }
    model.b2 -= scale * db2;
    model
}

fn serial_pipeline_step(model: TwoLayer, rate: f64) -> TwoLayer {
    let mut dw1 = [0.0; 2];
    let mut db1 = [0.0; 2];
    let mut dw2 = [0.0; 2];
    let mut db2 = 0.0;
    for &(x, y) in &PIPELINE_DATA {
        let h = [
            (model.w1[0] * x + model.b1[0]).tanh(),
            (model.w1[1] * x + model.b1[1]).tanh(),
        ];
        let residual = model.w2[0] * h[0] + model.w2[1] * h[1] + model.b2 - y;
        for j in 0..2 {
            dw2[j] += 2.0 * residual * h[j];
            let dz = 2.0 * residual * model.w2[j] * (1.0 - h[j].powi(2));
            dw1[j] += dz * x;
            db1[j] += dz;
        }
        db2 += 2.0 * residual;
    }
    apply_two_layer(model, dw1, db1, dw2, db2, rate)
}

fn pipeline_training_step(model: TwoLayer, rate: f64) -> Result<TwoLayer, &'static str> {
    if !rate.is_finite() || rate <= 0.0 {
        return Err("learning rate must be finite and positive");
    }
    let (forward_tx, forward_rx) = mpsc::channel();
    let (backward_tx, backward_rx) = mpsc::channel::<(usize, f64, [f64; 2], [f64; 2])>();
    let (first_gradient_tx, first_gradient_rx) = mpsc::channel();
    let (second_gradient_tx, second_gradient_rx) = mpsc::channel();
    thread::scope(|scope| -> Result<(), &'static str> {
        let first = scope.spawn(move || -> Result<(), &'static str> {
            for (id, &(x, y)) in PIPELINE_DATA.iter().enumerate() {
                let h = [
                    (model.w1[0] * x + model.b1[0]).tanh(),
                    (model.w1[1] * x + model.b1[1]).tanh(),
                ];
                forward_tx
                    .send((id, x, y, h))
                    .map_err(|_| "second stage closed")?;
            }
            drop(forward_tx);
            let mut dw1 = [0.0; 2];
            let mut db1 = [0.0; 2];
            for (_id, x, h, dh) in backward_rx {
                for j in 0..2 {
                    let dz = dh[j] * (1.0 - h[j].powi(2));
                    dw1[j] += dz * x;
                    db1[j] += dz;
                }
            }
            first_gradient_tx
                .send((dw1, db1))
                .map_err(|_| "coordinator closed")
        });
        let second = scope.spawn(move || -> Result<(), &'static str> {
            let mut dw2 = [0.0; 2];
            let mut db2 = 0.0;
            for (id, x, y, h) in forward_rx {
                let residual = model.w2[0] * h[0] + model.w2[1] * h[1] + model.b2 - y;
                for j in 0..2 {
                    dw2[j] += 2.0 * residual * h[j];
                }
                db2 += 2.0 * residual;
                backward_tx
                    .send((
                        id,
                        x,
                        h,
                        [2.0 * residual * model.w2[0], 2.0 * residual * model.w2[1]],
                    ))
                    .map_err(|_| "first stage closed")?;
            }
            second_gradient_tx
                .send((dw2, db2))
                .map_err(|_| "coordinator closed")
        });
        first.join().map_err(|_| "first stage panicked")??;
        second.join().map_err(|_| "second stage panicked")??;
        Ok(())
    })?;
    let (dw1, db1) = first_gradient_rx
        .recv()
        .map_err(|_| "missing first-stage gradients")?;
    let (dw2, db2) = second_gradient_rx
        .recv()
        .map_err(|_| "missing second-stage gradients")?;
    let next = apply_two_layer(model, dw1, db1, dw2, db2, rate);
    if next
        .w1
        .into_iter()
        .chain(next.b1)
        .chain(next.w2)
        .chain([next.b2])
        .any(|value| !value.is_finite())
    {
        return Err("pipeline update became nonfinite");
    }
    Ok(next)
}

impl TrainingState {
    fn train_step(
        mut self,
        data: &[Example],
        batch_size: usize,
        rate: f64,
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
        self.model = data_parallel_step(self.model, &batch, rate)?;
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
            self.model.w[0],
            self.model.w[1],
            self.model.w[2],
            self.model.w[3],
            self.model.b
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
                w: [numbers[0], numbers[1], numbers[2], numbers[3]],
                b: numbers[4],
            },
            step,
            cursor,
        })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let initial = TrainingState {
        model: Model {
            w: [0.0; 4],
            b: 0.0,
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
        tensor_parallel_predict(resumed.model, DATA[0].x)?
    );
    let tensor_update = tensor_parallel_step(resumed.model, &DATA[..5], 0.02)?;
    let serial_update = serial_step(resumed.model, &DATA[..5], 0.02)?;
    println!(
        "tensor-parallel update matches serial within 1e-12: {}",
        tensor_update
            .w
            .iter()
            .zip(serial_update.w)
            .all(|(a, b)| (a - b).abs() < 1e-12)
            && (tensor_update.b - serial_update.b).abs() < 1e-12
    );
    let losses = pipeline_losses(resumed.model, &DATA)?;
    println!(
        "pipeline stages communicated {} activations; mean loss {:.6}",
        losses.len(),
        losses.iter().sum::<f64>() / losses.len() as f64
    );
    let two_layer = TwoLayer {
        w1: [0.2, -0.3],
        b1: [0.1, 0.0],
        w2: [0.4, -0.2],
        b2: 0.0,
    };
    let trained_pipeline = pipeline_training_step(two_layer, 0.05)?;
    let serial_pipeline = serial_pipeline_step(two_layer, 0.05);
    println!(
        "pipeline backward updated both stages and matches serial: {}",
        trained_pipeline.w1 != two_layer.w1
            && trained_pipeline.w2 != two_layer.w2
            && trained_pipeline
                .w1
                .into_iter()
                .chain(trained_pipeline.b1)
                .chain(trained_pipeline.w2)
                .chain([trained_pipeline.b2])
                .zip(
                    serial_pipeline
                        .w1
                        .into_iter()
                        .chain(serial_pipeline.b1)
                        .chain(serial_pipeline.w2)
                        .chain([serial_pipeline.b2]),
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

    #[test]
    fn parallel_step_matches_serial_and_shards_match_dot() -> Result<(), &'static str> {
        let model = Model {
            w: [0.2, -0.1, 0.3, 0.4],
            b: -0.2,
        };
        let close = |a: Model, b: Model| {
            a.w.iter().zip(b.w).all(|(x, y)| (x - y).abs() < 1e-12) && (a.b - b.b).abs() < 1e-12
        };
        for size in [1, 3, 5] {
            assert!(close(
                data_parallel_step(model, &DATA[..size], 0.03)?,
                serial_step(model, &DATA[..size], 0.03)?
            ));
        }
        assert!(close(
            tensor_parallel_step(model, &DATA[..5], 0.03)?,
            serial_step(model, &DATA[..5], 0.03)?
        ));
        assert!(
            (tensor_parallel_predict(model, DATA[2].x)? - predict(model, &DATA[2].x)).abs() < 1e-12
        );
        Ok(())
    }

    #[test]
    fn pipeline_and_recovery_preserve_results() -> Result<(), &'static str> {
        let initial = TrainingState {
            model: Model {
                w: [0.0; 4],
                b: 0.0,
            },
            step: 0,
            cursor: 0,
        };
        let direct: Vec<_> = DATA
            .iter()
            .map(|e| (predict(initial.model, &e.x) - e.y).powi(2))
            .collect();
        assert_eq!(pipeline_losses(initial.model, &DATA)?, direct);
        let first = initial.train_step(&DATA, 3, 0.05)?;
        assert_eq!(first.cursor, 3);
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
            w1: [0.2, -0.3],
            b1: [0.1, 0.0],
            w2: [0.4, -0.2],
            b2: 0.0,
        };
        let parallel = pipeline_training_step(model, 0.05)?;
        let serial = serial_pipeline_step(model, 0.05);
        let loss = |m: TwoLayer| {
            PIPELINE_DATA
                .iter()
                .map(|&(x, y)| {
                    let prediction = m.w2[0] * (m.w1[0] * x + m.b1[0]).tanh()
                        + m.w2[1] * (m.w1[1] * x + m.b1[1]).tanh()
                        + m.b2;
                    (prediction - y).powi(2)
                })
                .sum::<f64>()
                / PIPELINE_DATA.len() as f64
        };
        let mut plus = model;
        let mut minus = model;
        plus.w1[0] += 1e-5;
        minus.w1[0] -= 1e-5;
        let numerical = (loss(plus) - loss(minus)) / 2e-5;
        let analytic = (model.w1[0] - parallel.w1[0]) / 0.05;
        assert!((analytic - numerical).abs() < 1e-6 + 1e-4 * numerical.abs());

        for (a, b) in parallel
            .w1
            .into_iter()
            .chain(parallel.b1)
            .chain(parallel.w2)
            .chain([parallel.b2])
            .zip(
                serial
                    .w1
                    .into_iter()
                    .chain(serial.b1)
                    .chain(serial.w2)
                    .chain([serial.b2]),
            )
        {
            assert!((a - b).abs() < 1e-12);
        }
        Ok(())
    }
}
