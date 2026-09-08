fn next_counts(data: &[u8], current: u8) -> [usize; 256] {
    // TODO: count each byte immediately after current.
    let _ = (data, current);
    todo!()
}
fn main() {
    println!("Count next bytes; embeddings will learn a smooth version.");
}
#[test]
fn counts_successors() {
    let c = next_counts(b"abacab", b'a');
    assert_eq!(c[b'b' as usize], 2);
    assert_eq!(c[b'c' as usize], 1);
}
