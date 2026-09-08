use ch36::{Config, Decoder};
fn model() -> Decoder {
    Decoder::new(
        Config {
            vocab_size: 8,
            context: 6,
            width: 8,
            heads: 2,
            layers: 1,
            ff_width: 16,
        },
        40,
    )
    .unwrap()
}
fn cached_last_logits(_model: &Decoder, _tokens: &[usize]) -> Vec<f32> {
    // TODO: append each layer's projected K and V once, then return the newest logits.
    todo!("implement incremental decoding")
}
fn main() {
    let _guided: fn(&Decoder, &[usize]) -> Vec<f32> = cached_last_logits;
    let m = model();
    let logits = m.forward(&[1, 2, 3]).unwrap();
    println!(
        "checkpoint: full decoder produced {} logits; last row starts {:.5}",
        logits.len(),
        logits[logits.len() - 8]
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_matches_full() {
        let m = model();
        let all = m.forward(&[1, 2, 3]).unwrap();
        assert_eq!(cached_last_logits(&m, &[1, 2, 3]), all[all.len() - 8..]);
    }
}
