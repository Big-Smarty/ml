//! Chapter 50 learner algorithms. Baseline weights each nonempty worker equally and freezes the first pipeline stage. Implement example-weighted reduction and full shard/pipeline updates using supplied threads and channels.
include!("common/ch50.rs");
include!("checks/ch50.rs");
fn aggregate_gradient_sums(parts: &[Gradient]) -> Result<Gradient, &'static str> {
    let active: Vec<_> = parts.iter().filter(|p| p.count > 0).collect();
    if active.is_empty() {
        return Err("at least one worker example is required");
    }
    // Worker-weighted mean objective; the goal instead gives every example equal weight.
    let mut total = Gradient {
        weights: [0.0; 4],
        bias: 0.0,
        count: active.len(),
    };
    for part in active {
        for (sum, value) in total.weights.iter_mut().zip(part.weights) {
            *sum += value / part.count as f64;
        }
        total.bias += part.bias / part.count as f64;
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
    let (_hidden_weight_sums, _hidden_bias_sums) = first_gradient_rx
        .recv()
        .map_err(|_| "missing first-stage gradients")?;
    let (output_weight_sums, output_bias_sum) = second_gradient_rx
        .recv()
        .map_err(|_| "missing second-stage gradients")?;
    apply_two_layer(
        model,
        [0.0; 2],
        [0.0; 2],
        output_weight_sums,
        output_bias_sum,
        data.len(),
        learning_rate,
    )
}
