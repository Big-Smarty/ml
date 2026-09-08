fn bytes(text: &str) -> Vec<u8> {
    text.as_bytes().to_vec()
}
fn merge_pair(token_ids: &[u16], _pair: (u16, u16), _new_token_id: u16) -> Vec<u16> {
    // TODO: replace every non-overlapping matching pair.
    let _ = token_ids;
    todo!("guided repair: implement non-overlapping pair merging")
}
fn main() {
    let _exercise = merge_pair as fn(&[u16], (u16, u16), u16) -> Vec<u16>;
    let text = "café 咖啡";
    println!(
        "{text:?} has {} Unicode scalar values and {} UTF-8 bytes",
        text.chars().count(),
        bytes(text).len()
    );
    println!("Complete merge_pair, then run cargo test.");
}
#[test]
fn merges_non_overlapping_pairs() {
    assert_eq!(merge_pair(&[1, 2, 1, 2, 2], (1, 2), 9), vec![9, 9, 2]);
}
