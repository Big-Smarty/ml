fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut output = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            output[row * n + col] = (0..k)
                .map(|inner| a[row * k + inner] * b[inner * n + col])
                .sum();
        }
    }
    output
}

fn tile_count(_k: usize) -> usize {
    // TODO: include the partial K tile.
    todo!("include the partial K tile")
}

fn main() {
    let _guided = tile_count;
    println!(
        "CPU checkpoint: {:?}",
        cpu_matmul(&[1.0, 2.0], &[3.0, 4.0, 5.0, 6.0], 1, 2, 2)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_inner_dimension_keeps_last_tile() {
        assert_eq!(
            tile_count(17),
            2,
            "guided repair: floor division drops the partial K tile"
        );
    }
}
