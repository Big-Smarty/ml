//! Workers borrow read-only parameters. Only the output chunks are mutable; all joins are observed.
use crate::{
    common::{self, Shape},
    training::{self, Gradient, Model},
};
use std::thread::{self, ThreadId};
pub fn parallel_dense(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    y: &mut [f32],
    s: Shape,
    threads: usize,
) -> Result<Vec<ThreadId>, String> {
    common::validate(x, w, bias, y, s)?;
    if threads == 0 || threads > 64 {
        return Err("request 1..64 threads".into());
    }
    let rows = s.m.div_ceil(threads.min(s.m));
    // ponytail: create OS threads per call; use a persistent pool only after launch cost matters.
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for (input, output) in x.chunks(rows * s.k).zip(y.chunks_mut(rows * s.n)) {
            handles.push(
                thread::Builder::new()
                    .spawn_scoped(scope, move || {
                        common::dense(
                            input,
                            w,
                            bias,
                            output,
                            Shape {
                                m: input.len() / s.k,
                                ..s
                            },
                        )?;
                        Ok(thread::current().id())
                    })
                    .map_err(|e| format!("spawn failed: {e}"))?,
            );
        }
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| "inference worker panicked".to_string())?
            })
            .collect()
    })
}
pub fn loss_parallel(
    model: &Model,
    x: &[f32],
    targets: &[f32],
    k: usize,
    threads: usize,
) -> Result<(f64, Gradient, Vec<ThreadId>), String> {
    let batch = training::validate_training_data(x, targets, k)?;
    model.validate_inputs(x, k)?;
    if threads == 0 || threads > 64 {
        return Err("request 1..64 threads".into());
    }
    let rows = batch.div_ceil(threads.min(batch));
    let partials = thread::scope(|scope| {
        let mut handles = Vec::new();
        for (input, target) in x.chunks(rows * k).zip(targets.chunks(rows)) {
            handles.push(
                thread::Builder::new()
                    .spawn_scoped(scope, move || {
                        let (value, g) = training::loss_and_gradient_sum(model, input, target, k);
                        (value, g, thread::current().id())
                    })
                    .map_err(|e| format!("spawn failed: {e}"))?,
            );
        }
        handles
            .into_iter()
            .map(|h| h.join().map_err(|_| "gradient worker panicked".to_string()))
            .collect::<Result<Vec<_>, String>>()
    })?;
    let mut total = Gradient {
        weights: vec![0.; k],
        bias: 0.,
    };
    let mut value = 0.;
    let mut ids = Vec::new();
    // Handle order fixes grouping regardless of completion order. Never average individual shards.
    for (partial, g, id) in partials {
        value += partial;
        total.bias += g.bias;
        for (sum, term) in total.weights.iter_mut().zip(g.weights) {
            *sum += term;
        }
        ids.push(id);
    }
    let (value, g) = training::mean_loss_and_gradient(value, total, batch)?;
    Ok((value, g, ids))
}
pub fn dense(x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32], s: Shape) -> Result<(), String> {
    parallel_dense(x, w, bias, y, s, 3)?;
    Ok(())
}
pub fn run(args: &[String]) -> Result<(), String> {
    common::demo(dense)?;
    crate::ch27::train(loss_parallel)?;
    if args.iter().any(|a| a == "--bench") {
        common::benchmark(
            "scoped dense threads=3; includes launch/join and receipt allocation",
            dense,
        )?;
    }
    Ok(())
}
pub fn check() -> Result<(), String> {
    crate::ch27::verify(parallel_dense, loss_parallel)
}
