fn top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut ids: Vec<_> = (0..logits.len()).collect();
    ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    ids.truncate(k.min(ids.len()));
    ids
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
