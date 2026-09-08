fn merge(ids: &[u16], pair: (u16, u16), new_id: u16) -> Vec<u16> {
    // TODO: merge non-overlapping occurrences of pair.
    let _ = (pair, new_id);
    let _ = ids;
    todo!()
}
fn main() {
    println!("BPE begins with bytes and merges pairs.");
}
#[test]
fn merges_pairs() {
    assert_eq!(merge(&[1, 2, 1, 2, 2], (1, 2), 9), vec![9, 9, 2]);
}
