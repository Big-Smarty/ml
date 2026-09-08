#[cfg(test)]
use std::thread;
fn predict(w: &[f32], x: &[f32]) -> f32 {
    w.iter().zip(x).map(|(a, b)| a * b).sum()
}
fn sequential_mse(w: &[f32], x: &[f32], y: &[f32]) -> f64 {
    x.chunks_exact(w.len())
        .zip(y)
        .map(|(r, t)| {
            let e = predict(w, r) as f64 - *t as f64;
            e * e
        })
        .sum::<f64>()
        / y.len() as f64
}
#[cfg(test)]
fn parallel_mse(_w: &[f32], _x: &[f32], _y: &[f32], _threads: usize) -> f64 {
    // TODO: give each scoped thread a disjoint row range, then reduce partial sums in handle order.
    thread::scope(|_| {});
    todo!("implement fixed-order shard reduction")
}
fn main() {
    let x = [1., 2., 2., 1.];
    let y = [5., 4.];
    let w = [1., 2.];
    println!("prior scalar inference MSE={}", sequential_mse(&w, &x, &y));
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parallel_reduction_matches_scalar() {
        let x = [1., 2., 2., 1., 3., -1.];
        let y = [5., 4., 2.];
        let w = [1., 2.];
        assert!((parallel_mse(&w, &x, &y, 2) - sequential_mse(&w, &x, &y)).abs() < 1e-12);
    }
}
