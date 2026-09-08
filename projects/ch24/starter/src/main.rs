fn rnn_step(x: f64, previous: f64, wx: f64, wh: f64) -> f64 {
    (wx * x + wh * previous).tanh()
}
fn preference(_user: &[f64], _item: &[f64]) -> f64 {
    // TODO: return the embedding dot product.
    todo!("sum matching coordinate products")
}
fn main() {
    let _guided_step = preference;
    println!(
        "previous checkpoint recurrent state: {:.3}",
        rnn_step(1.0, 0.5, 0.7, 0.2)
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn embedding_score_is_dot_product() {
        assert_eq!(preference(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
    }
}
