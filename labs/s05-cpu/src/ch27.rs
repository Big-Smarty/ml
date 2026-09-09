//! Working serial inference/training. Implement disjoint scoped inference and private gradient shards.
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
    if threads == 0 || threads > 64 {
        return Err("request 1..64 threads".into());
    }
    common::dense(x, w, bias, y, s)?;
    Ok(vec![thread::current().id()])
}
pub fn loss_parallel(
    model: &Model,
    x: &[f32],
    targets: &[f32],
    k: usize,
    threads: usize,
) -> Result<(f64, Gradient, Vec<ThreadId>), String> {
    if threads == 0 || threads > 64 {
        return Err("request 1..64 threads".into());
    }
    let (loss, gradient) = training::loss_and_gradient(model, x, targets, k)?;
    Ok((loss, gradient, vec![thread::current().id()]))
}
pub type Parallel =
    fn(&[f32], &[f32], &[f32], &mut [f32], Shape, usize) -> Result<Vec<ThreadId>, String>;
pub type Loss =
    fn(&Model, &[f32], &[f32], usize, usize) -> Result<(f64, Gradient, Vec<ThreadId>), String>;
pub fn dense(x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32], s: Shape) -> Result<(), String> {
    parallel_dense(x, w, bias, y, s, 3)?;
    Ok(())
}
fn near(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-6 + 1e-5 * b.abs()
}
pub fn verify(parallel: Parallel, loss: Loss) -> Result<(), String> {
    for (m, k, n, t) in [(7, 5, 3, 3), (2, 17, 1, 8), (1, 3, 2, 1), (17, 3, 4, 4)] {
        let s = Shape { m, k, n };
        let (x, w, bias) = (
            common::values(m * k, 11),
            common::values(n * k, 47),
            common::values(n, 19),
        );
        let mut y = vec![99.; m * n];
        let ids = parallel(&x, &w, &bias, &mut y, s, t)?;
        common::close(&y, &common::oracle(&x, &w, &bias, s), k)?;
        let expected = m.div_ceil(m.div_ceil(t.min(m)));
        verify_workers(&ids, expected)?;
    }
    for (batch, k, t) in [(31, 4, 3), (17, 3, 4), (2, 2, 8)] {
        let (x, y) = training::fixture(batch, k);
        let model = Model {
            weights: common::values(k, 9),
            bias: 0.3,
        };
        let reference = training::loss_and_gradient(&model, &x, &y, k)?;
        let (value, g, ids) = loss(&model, &x, &y, k, t)?;
        verify_workers(&ids, batch.div_ceil(batch.div_ceil(t.min(batch))))?;
        if g.weights.len() != k
            || !near(value, reference.0)
            || !near(g.bias, reference.1.bias)
            || !g
                .weights
                .iter()
                .zip(&reference.1.weights)
                .all(|(&a, &b)| near(a, b))
        {
            return Err(
                "GOAL_NOT_MET: parallel loss/gradient must sum shards then divide once by complete batch".into(),
            );
        }
    }
    let mut model = Model {
        weights: vec![0.5, -1.],
        bias: 0.25,
    };
    let (value, g, _) = loss(&model, &[1., 2., 3., 1.], &[-1., 1.], 2, 8)?;
    if g.weights.len() != 2
        || !near(value, 0.0625)
        || !near(g.weights[0], -1.)
        || !near(g.weights[1], -0.75)
        || !near(g.bias, -0.5)
    {
        return Err("GOAL_NOT_MET: hand gradient differs".into());
    }
    update(&mut model, &g, 0.1)?;
    if !near(model.weights[0] as f64, 0.6) || !near(model.bias as f64, 0.3) {
        return Err("GOAL_NOT_MET: simultaneous update differs".into());
    }
    train(loss)?;
    Ok(())
}
fn verify_workers(ids: &[ThreadId], expected: usize) -> Result<(), String> {
    if ids.len() != expected
        || ids.contains(&thread::current().id())
        || ids.iter().collect::<std::collections::HashSet<_>>().len() != expected
    {
        return Err(format!("GOAL_REVIEW_REQUIRED: expected {expected} distinct scoped workers, observed {} receipts (serial baseline uses calling thread); implement row partitioning",ids.len()));
    }
    Ok(())
}
pub fn update(model: &mut Model, g: &Gradient, rate: f32) -> Result<(), String> {
    if !rate.is_finite() || rate <= 0. || model.weights.len() != g.weights.len() {
        return Err("invalid learning rate or gradient shape".into());
    }
    let bias = model.bias - rate * g.bias as f32;
    let weights: Vec<_> = model
        .weights
        .iter()
        .zip(&g.weights)
        .map(|(&w, &dw)| w - rate * dw as f32)
        .collect();
    if !bias.is_finite() || weights.iter().any(|v| !v.is_finite()) {
        return Err("update overflow; model unchanged".into());
    }
    model.weights = weights;
    model.bias = bias;
    Ok(())
}
pub fn train(loss: Loss) -> Result<(), String> {
    let (batch, features, threads, steps, rate) = (31, 4, 3, 60, 0.2);
    let (x, y) = training::fixture(batch, features);
    let mut model = Model {
        weights: vec![0.; features],
        bias: 0.,
    };
    let before = loss(&model, &x, &y, features, threads)?.0;
    for _ in 0..steps {
        let (_, g, _) = loss(&model, &x, &y, features, threads)?;
        update(&mut model, &g, rate)?;
    }
    let after = loss(&model, &x, &y, features, threads)?.0;
    println!("{batch} rows, {features} features, requested {threads} workers, {steps} steps lr={rate}: training MSE {before:.6} -> {after:.6}; synthetic mechanics, not held-out evaluation");
    let predictions = model.predict_batch(&x, features)?;
    println!(
        "first predictions={:?}",
        &predictions[..3.min(predictions.len())]
    );
    if after >= before * 1e-3 {
        return Err("GOAL_NOT_MET: training did not lower MSE by required factor".into());
    }
    Ok(())
}
pub fn run(args: &[String]) -> Result<(), String> {
    common::demo(dense)?;
    train(loss_parallel)?;
    if args.iter().any(|a| a == "--bench") {
        common::benchmark("serial baseline requested threads=3, actual=1", dense)?;
    }
    Ok(())
}
pub fn check() -> Result<(), String> {
    verify(parallel_dense, loss_parallel)?;
    Err("GOAL_REVIEW_REQUIRED: numerics and worker receipts pass; inspect disjoint slices, private gradient sums, creation-order joins, and one full-batch division".into())
}
#[cfg(test)]
mod tests {
    #[test]
    fn baseline_and_parallel() -> Result<(), String> {
        super::run(&[])?;
        crate::solutions::ch27::check()
    }
    #[test]
    fn invalid_update_is_atomic() {
        let mut model = crate::training::Model {
            weights: vec![0.],
            bias: 0.,
        };
        let g = crate::training::Gradient {
            weights: vec![f64::MAX],
            bias: 1.,
        };
        assert!(super::update(&mut model, &g, 1.).is_err());
        assert_eq!(model.weights, vec![0.]);
        assert_eq!(model.bias, 0.);
    }
}

#[cfg(test)]
mod boundary_checks {
    use super::*;
    #[test]
    fn both_training_paths_reject_invalid_data() {
        let model = Model {
            weights: vec![1.],
            bias: 0.,
        };
        for loss in [loss_parallel as Loss, crate::solutions::ch27::loss_parallel] {
            assert!(loss(&model, &[], &[], 1, 2).is_err());
            assert!(loss(&model, &[1.], &[0.], 1, 0).is_err());
            assert!(loss(&model, &[1.], &[0.], 1, 65).is_err());
            assert!(loss(&model, &[f32::NAN], &[0.], 1, 2).is_err());
            assert!(loss(&model, &[1.], &[], 1, 2).is_err());
        }
        assert!(model
            .predict_batch(&[], 1)
            .expect("empty prediction is valid")
            .is_empty());
    }
}
