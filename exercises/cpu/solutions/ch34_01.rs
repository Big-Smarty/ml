fn merge(ids: &[u16], pair: (u16, u16), new_id: u16) -> Vec<u16> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
            out.push(new_id);
            i += 2
        } else {
            out.push(ids[i]);
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
    assert_eq!(merge(&[1, 2, 1, 2, 2], (1, 2), 9), vec![9, 9, 2]);
}
