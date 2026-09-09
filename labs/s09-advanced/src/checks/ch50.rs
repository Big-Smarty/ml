pub fn check() -> Result<(), String> {
    let parts = [
        Gradient {
            weights: [4.0; 4],
            bias: 4.0,
            count: 1,
        },
        Gradient {
            weights: [6.0; 4],
            bias: 6.0,
            count: 3,
        },
    ];
    let g = aggregate_gradient_sums(&parts)?;
    crate::ensure(
        g.count == 4 && g.weights == [10.0; 4] && g.bias == 10.0,
        &format!("sum/count reduction got {g:?}; expected sum 10 and count 4, giving mean 2.5"),
    )?;
    let model = Model {
        weights: [0.2, -0.1, 0.3, 0.4],
        bias: -0.2,
    };
    for size in [1, 3, 5] {
        let oracle = serial_step(model, &DATA[..size], 0.03)?;
        for actual in [
            data_parallel_step(model, &DATA[..size], 0.03)?,
            tensor_parallel_step(model, &DATA[..size], 0.03)?,
        ] {
            crate::ensure(
                actual
                    .weights
                    .iter()
                    .zip(oracle.weights)
                    .all(|(&a, b)| crate::close(a, b))
                    && crate::close(actual.bias, oracle.bias),
                "real worker update differs from serial on uneven shards",
            )?;
        }
    }
    let two = TwoLayer {
        hidden_weights: [0.2, -0.3],
        hidden_biases: [0.1, 0.0],
        output_weights: [0.4, -0.2],
        output_bias: 0.0,
    };
    let actual = pipeline_training_step(two, &PIPELINE_DATA[..3], 0.05)?;
    let oracle = serial_pipeline_step(two, &PIPELINE_DATA[..3], 0.05)?;
    crate::ensure(
        actual
            .hidden_weights
            .iter()
            .zip(oracle.hidden_weights)
            .all(|(&a, b)| crate::close(a, b))
            && actual.hidden_weights != two.hidden_weights,
        "pipeline backward must update the first stage using old output weights",
    )?;
    let initial = TrainingState {
        model,
        step: 0,
        cursor: 0,
    };
    let first = initial.train_step(&DATA, 3, 0.05)?;
    crate::ensure(
        TrainingState::decode(&first.encode())?.train_step(&DATA, 3, 0.05)?
            == first.train_step(&DATA, 3, 0.05)?,
        "resume must reproduce the next update",
    )
}

pub fn run(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("this experiment takes no extra arguments".into());
    }
    demo().map_err(|e| e.to_string())
}
