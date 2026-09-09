//! Worked solution: Combine gradient sums and example counts before averaging. Tensor shards share a complete residual, and pipeline stages send activation gradients backward while using frozen parameters. Apply both stages only after all messages are collected. Recovery restores model, step, and next minibatch cursor.
//! Compare intermediate quantities with the lesson before using the final goal check.
//! Worked solution for chapter 50. Read the comments and lesson explanations before comparing.
include!("../common/ch50.rs");
include!("../checks/ch50.rs");
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
