fn reduce_in_order(partials: &[Vec<f64>]) -> Vec<f64> {
    let mut total = vec![0.0; partials.first().map_or(0, Vec::len)];
    for shard in partials {
        for (sum, value) in total.iter_mut().zip(shard) {
            *sum += value;
        }
    }
    total
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
