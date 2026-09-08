use ch36::{Config, Decoder};
fn model() -> Decoder {
    Decoder::new(
        Config {
            vocab_size: 8,
            context: 4,
            width: 8,
            heads: 2,
            layers: 1,
            ff_width: 16,
        },
        43,
    )
    .unwrap()
}
fn low_rank_delta(_a: &[f32], _b: &[f32], _rows: usize, _rank: usize, _cols: usize) -> Vec<f32> {
    // TODO: return A times B in row-major order.
    todo!("implement the LoRA product")
}
fn main() {
    let _guided = std::hint::black_box(low_rank_delta);
    let mut m = model();
    let input = [0, 1, 2];
    let targets = [1, 2, 3];
    let initial = m.loss_and_grad(&input, &targets).unwrap().loss;
    for _ in 0..30 {
        let g = m.loss_and_grad(&input, &targets).unwrap();
        m.apply_sgd(&g, 0.06).unwrap();
    }
    let trained = m.loss_and_grad(&input, &targets).unwrap().loss;
    println!("checkpoint pretrained decoder loss: {initial:.4} -> {trained:.4}");
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rank_one_product() {
        assert_eq!(
            low_rank_delta(&[1.0, 2.0], &[3.0, 4.0], 2, 1, 2),
            [3.0, 4.0, 6.0, 8.0]
        );
    }
}
