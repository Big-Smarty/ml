fn reduce_in_order(partials: &[Vec<f64>]) -> Vec<f64> {
    // TODO: add each shard in slice order into one zeroed gradient.
    let _ = partials;
    todo!("implement deterministic reduction")
}

fn main() {
    println!("{:?}", reduce_in_order(&[vec![1., 2.], vec![3., 4.]]));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_private_gradient_shards() {
        assert_eq!(reduce_in_order(&[vec![1., 2.], vec![3., 4.]]), vec![4., 6.]);
    }
}
