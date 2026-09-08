fn reduce_gradient_sums_in_order(shard_weight_gradients: &[Vec<f64>]) -> Vec<f64> {
    // TODO: add each worker's Gradient.weights sum in slice order; the caller averages afterward.
    let _ = shard_weight_gradients;
    todo!("implement deterministic reduction")
}

fn main() {
    println!(
        "{:?}",
        reduce_gradient_sums_in_order(&[vec![1., 2.], vec![3., 4.]])
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_private_gradient_shards() {
        assert_eq!(
            reduce_gradient_sums_in_order(&[vec![1., 2.], vec![3., 4.]]),
            vec![4., 6.]
        );
    }
}
