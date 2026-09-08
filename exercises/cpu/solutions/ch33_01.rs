fn next_counts(data: &[u8], current: u8) -> [usize; 256] {
    let mut counts = [0; 256];
    for pair in data.windows(2) {
        if pair[0] == current {
            counts[pair[1] as usize] += 1;
        }
    }
    counts
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
