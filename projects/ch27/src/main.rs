//! Deterministic sharded linear-regression training and inference with scoped threads.
use std::thread;

#[derive(Clone, Debug)]
struct Model {
    weights: Vec<f32>,
    bias: f32,
}

fn fixture(rows: usize, features: usize) -> (Vec<f32>, Vec<f32>) {
    let mut x = Vec::with_capacity(rows * features);
    let mut y = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut target = 0.25;
        for f in 0..features {
            let v = (((r * 17 + f * 13) % 29) as f32 - 14.0) / 14.0;
            x.push(v);
            target += v * (f as f32 + 1.0) * 0.3;
        }
        y.push(target);
    }
    (x, y)
}

fn check(x: &[f32], y: &[f32], features: usize) -> Result<usize, String> {
    if features == 0
        || !x.len().is_multiple_of(features)
        || x.len() / features != y.len()
        || y.is_empty()
    {
        return Err("expected nonempty x=[rows,features] and y=[rows]".into());
    }
    if x.iter().chain(y).any(|v| !v.is_finite()) {
        return Err("data must be finite".into());
    }
    Ok(y.len())
}

impl Model {
    fn check_input(&self, x: &[f32], features: usize) -> Result<(), String> {
        if features == 0 || self.weights.len() != features || !x.len().is_multiple_of(features) {
            return Err("inference shape mismatch".into());
        }
        if !self.bias.is_finite() || self.weights.iter().chain(x).any(|v| !v.is_finite()) {
            return Err("model and input must be finite".into());
        }
        Ok(())
    }
    fn predict_row(&self, row: &[f32]) -> f32 {
        self.bias
            + self
                .weights
                .iter()
                .zip(row)
                .map(|(w, x)| w * x)
                .sum::<f32>()
    }
    fn infer(&self, x: &[f32], features: usize) -> Result<Vec<f32>, String> {
        self.check_input(x, features)?;
        let out: Vec<_> = x
            .chunks_exact(features)
            .map(|r| self.predict_row(r))
            .collect();
        if out.iter().any(|v| !v.is_finite()) {
            return Err("prediction overflow; rescale the inputs".into());
        }
        Ok(out)
    }
    fn infer_parallel(
        &self,
        x: &[f32],
        features: usize,
        threads: usize,
    ) -> Result<Vec<f32>, String> {
        self.check_input(x, features)?;
        if threads == 0 {
            return Err("inference shape or thread count is invalid".into());
        }
        let rows = x.len() / features;
        if rows == 0 {
            return Ok(Vec::new());
        }
        let shards = threads.min(rows);
        let per = rows.div_ceil(shards);
        let mut out = vec![0.; rows];
        thread::scope(|scope| {
            let mut tail = &mut out[..];
            for shard in 0..shards {
                let start = shard * per;
                if start >= rows {
                    break;
                }
                let end = start.saturating_add(per).min(rows);
                let len = end - start;
                let (now, rest) = tail.split_at_mut(len);
                tail = rest;
                let input = &x[start * features..end * features];
                scope.spawn(move || {
                    for (r, dst) in input.chunks_exact(features).zip(now) {
                        *dst = self.predict_row(r);
                    }
                });
            }
        });
        if out.iter().any(|v| !v.is_finite()) {
            return Err("prediction overflow; rescale the inputs".into());
        }
        Ok(out)
    }
}

fn shard_gradient(
    model: &Model,
    x: &[f32],
    y: &[f32],
    features: usize,
    start: usize,
    end: usize,
) -> (f64, Vec<f64>, f64) {
    let mut loss = 0.;
    let mut dw = vec![0.; features];
    let mut db = 0.;
    for r in start..end {
        let row = &x[r * features..(r + 1) * features];
        let e = model.predict_row(row) as f64 - y[r] as f64;
        loss += e * e;
        db += 2. * e;
        for f in 0..features {
            dw[f] += 2. * e * row[f] as f64;
        }
    }
    (loss, dw, db)
}

fn gradient_parallel(
    model: &Model,
    x: &[f32],
    y: &[f32],
    features: usize,
    threads: usize,
) -> Result<(f64, Vec<f64>, f64), String> {
    let rows = check(x, y, features)?;
    model.check_input(x, features)?;
    if model.weights.len() != features || threads == 0 {
        return Err("model shape or thread count is invalid".into());
    }
    let shards = threads.min(rows);
    let per = rows.div_ceil(shards);
    let partials = thread::scope(|scope| {
        let mut handles = Vec::new();
        for shard in 0..shards {
            let start = shard * per;
            let end = start.saturating_add(per).min(rows);
            if start < end {
                handles
                    .push(scope.spawn(move || shard_gradient(model, x, y, features, start, end)));
            }
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("gradient worker panicked"))
            .collect::<Vec<_>>()
    });
    let (mut loss, mut dw, mut db) = (0., vec![0.; features], 0.);
    for (l, g, b) in partials {
        loss += l;
        db += b;
        for (i, v) in g.into_iter().enumerate() {
            dw[i] += v;
        }
    }
    let scale = 1. / rows as f64;
    for v in &mut dw {
        *v *= scale
    }
    if !loss.is_finite() || !db.is_finite() || dw.iter().any(|v| !v.is_finite()) {
        return Err("gradient overflow; rescale inputs or model".into());
    }
    Ok((loss * scale, dw, db * scale))
}

fn train_step(
    model: &mut Model,
    x: &[f32],
    y: &[f32],
    features: usize,
    threads: usize,
    rate: f32,
) -> Result<f64, String> {
    if !rate.is_finite() || rate <= 0. {
        return Err("learning rate must be finite and positive".into());
    }
    let (loss, dw, db) = gradient_parallel(model, x, y, features, threads)?;
    if model
        .weights
        .iter()
        .zip(&dw)
        .any(|(w, g)| !(w - rate * *g as f32).is_finite())
        || !(model.bias - rate * db as f32).is_finite()
    {
        return Err("update overflow; reduce the learning rate".into());
    }
    for (w, g) in model.weights.iter_mut().zip(dw) {
        *w -= rate * g as f32;
    }
    model.bias -= rate * db as f32;
    Ok(loss)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let features = 4;
    let (x, y) = fixture(64, features);
    let mut model = Model {
        weights: vec![0.; features],
        bias: 0.,
    };
    let before = gradient_parallel(&model, &x, &y, features, 4)?.0;
    for _ in 0..40 {
        train_step(&mut model, &x, &y, features, 4, 0.2)?;
    }
    let after = gradient_parallel(&model, &x, &y, features, 4)?.0;
    let predictions = model.infer_parallel(&x[..8 * features], features, 4)?;
    let scalar_predictions = model.infer(&x[..8 * features], features)?;
    if predictions != scalar_predictions {
        return Err("parallel inference disagrees with scalar inference".into());
    }
    println!("four-shard training MSE: {before:.6} -> {after:.6}");
    println!("model weights={:?} bias={:.4}", model.weights, model.bias);
    println!("parallel inference first 3={:?}", &predictions[..3]);
    println!("fixed shard boundaries and reduction order are reproducible for this build; other thread counts may change low bits");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gradient_and_update_match_nonzero_hand_example() {
        let mut model = Model {
            weights: vec![0.5, -1.0],
            bias: 0.25,
        };
        let x = [1.0, 2.0, 3.0, 1.0];
        let y = [-1.0, 1.0];
        let (loss, dw, db) = gradient_parallel(&model, &x, &y, 2, 8).unwrap();
        assert!(close(loss, 0.0625));
        assert!(close(dw[0], -1.0));
        assert!(close(dw[1], -0.75));
        assert!(close(db, -0.5));
        train_step(&mut model, &x, &y, 2, 8, 0.1).unwrap();
        assert!(close(model.weights[0] as f64, 0.6));
        assert!(close(model.weights[1] as f64, -0.925));
        assert!(close(model.bias as f64, 0.3));
    }
    #[test]
    fn invalid_arithmetic_is_rejected_before_committing_update() {
        let mut model = Model {
            weights: vec![0.0],
            bias: 0.0,
        };
        assert!(model.infer(&[f32::NAN], 1).is_err());
        assert!(model.infer_parallel(&[f32::NAN], 1, 2).is_err());
        assert!(train_step(&mut model, &[1.0], &[2.0], 1, 2, f32::MAX).is_err());
        assert_eq!(model.weights, vec![0.0]);
        assert_eq!(model.bias, 0.0);
        model.weights[0] = f32::MAX;
        assert!(gradient_parallel(&model, &[2.0], &[0.0], 1, 1).is_err());
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-6 + 1e-5 * b.abs()
    }
    #[test]
    fn parallel_inference_and_gradients_agree_with_scalar() {
        let f = 3;
        let (x, y) = fixture(17, f);
        let m = Model {
            weights: vec![0.2, -0.1, 0.4],
            bias: 0.3,
        };
        let a = m.infer(&x, f).unwrap();
        let b = m.infer_parallel(&x, f, 4).unwrap();
        assert_eq!(a, b);
        let scalar = shard_gradient(&m, &x, &y, f, 0, y.len());
        let p = gradient_parallel(&m, &x, &y, f, 4).unwrap();
        assert!(close(p.0, scalar.0 / y.len() as f64));
        for (i, g) in p.1.iter().enumerate() {
            assert!(close(*g, scalar.1[i] / y.len() as f64));
        }
        assert!(close(p.2, scalar.2 / y.len() as f64));
    }
    #[test]
    fn parallel_training_reduces_loss() {
        let f = 4;
        let (x, y) = fixture(31, f);
        let mut m = Model {
            weights: vec![0.; f],
            bias: 0.,
        };
        let before = gradient_parallel(&m, &x, &y, f, 3).unwrap().0;
        for _ in 0..60 {
            train_step(&mut m, &x, &y, f, 3, 0.2).unwrap();
        }
        assert!(gradient_parallel(&m, &x, &y, f, 3).unwrap().0 < before * 1e-3);
    }
    #[test]
    fn empty_inference_is_valid_but_empty_training_is_not() {
        let m = Model {
            weights: vec![1.],
            bias: 0.,
        };
        assert!(m.infer_parallel(&[], 1, 2).unwrap().is_empty());
        assert!(gradient_parallel(&m, &[], &[], 1, 2).is_err());
    }
    #[test]
    fn inference_handles_every_short_uneven_partition() {
        let m = Model {
            weights: vec![0.5, -0.25],
            bias: 0.1,
        };
        for rows in 1..20 {
            let (x, _) = fixture(rows, 2);
            let scalar = m.infer(&x, 2).unwrap();
            for threads in 1..25 {
                assert_eq!(m.infer_parallel(&x, 2, threads).unwrap(), scalar);
            }
        }
    }
}
