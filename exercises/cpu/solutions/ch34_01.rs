fn merge_pair(token_ids: &[u16], pair: (u16, u16), new_token_id: u16) -> Vec<u16> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < token_ids.len() {
        if i + 1 < token_ids.len() && (token_ids[i], token_ids[i + 1]) == pair {
            out.push(new_token_id);
            i += 2
        } else {
            out.push(token_ids[i]);
            i += 1
        }
    }
    out
}
fn main() {
    println!("BPE begins with bytes and merges pairs.");
}
#[test]
fn merges_pairs() {
    assert_eq!(merge_pair(&[1, 2, 1, 2, 2], (1, 2), 9), vec![9, 9, 2]);
}
