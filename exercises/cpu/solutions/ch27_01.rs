fn reduce_gradient_sums_in_order(shard_weight_gradients: &[Vec<f64>]) -> Vec<f64> {
    let mut total = vec![0.0; shard_weight_gradients.first().map_or(0, Vec::len)];
    for shard in shard_weight_gradients {
        for (sum, value) in total.iter_mut().zip(shard) {
            *sum += value;
        }
    }
    total
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
