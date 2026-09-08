fn top_k(_logits: &[f32], _k: usize) -> Vec<usize> {
    // TODO: return token IDs for the k largest logits, largest logit first.
    todo!("rank logits")
}
fn main() {
    println!("top two: {:?}", top_k(&[1.0, 4.0, 2.0], 2));
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selects_largest() {
        assert_eq!(top_k(&[1.0, 4.0, 2.0], 2), [1, 2]);
    }
}
