fn next_counts(data: &[u8], current: u8) -> [usize; 256] {
    // TODO: count each byte immediately after current.
    let _ = (data, current);
    todo!()
}
fn main() {
    println!("Count next-byte class IDs; the model will learn their probabilities.");
}
#[test]
fn counts_successors() {
    let counts = next_counts(b"abacab", b'a');
    assert_eq!(counts[b'b' as usize], 2);
    assert_eq!(counts[b'c' as usize], 1);
}
