fn normalized(values: &[f64]) -> Vec<f64> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|x| x - mean).collect()
}
fn encode(_features: &[f64; 4], _encoder_weights: &[[f64; 4]; 2]) -> [f64; 2] {
    // TODO: compute two weighted sums, one per bottleneck coordinate.
    todo!("matrix-vector product")
}
fn main() {
    let _guided_step = encode;
    println!(
        "previous checkpoint centered image: {:?}",
        normalized(&[1.0, 2.0, 3.0])
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bottleneck_is_a_matrix_vector_product() {
        assert_eq!(
            encode(
                &[1.0, 2.0, 3.0, 4.0],
                &[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]
            ),
            [1.0, 3.5]
        );
    }
}
