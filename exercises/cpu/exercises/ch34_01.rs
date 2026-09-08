fn merge_pair(token_ids: &[u16], pair: (u16, u16), new_token_id: u16) -> Vec<u16> {
    // TODO: merge non-overlapping occurrences of pair.
    let _ = (pair, new_token_id);
    let _ = token_ids;
    todo!()
}
fn main() {
    println!("BPE begins with bytes and merges pairs.");
}
#[test]
fn merges_pairs() {
    assert_eq!(merge_pair(&[1, 2, 1, 2, 2], (1, 2), 9), vec![9, 9, 2]);
}
