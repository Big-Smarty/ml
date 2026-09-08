use std::collections::HashMap;
fn byte_bigrams(data: &[u8]) -> HashMap<(u8, u8), usize> {
    let mut counts = HashMap::new();
    for pair in data.windows(2) {
        *counts.entry((pair[0], pair[1])).or_default() += 1;
    }
    counts
}

fn most_likely_after(_data: &[u8], _current: u8) -> Option<u8> {
    // TODO: count successors and choose the most frequent byte.
    todo!("guided repair: implement the next-byte count")
}
fn main() {
    let _exercise = most_likely_after as fn(&[u8], u8) -> Option<u8>;
    let text = b"rust runs. rust learns.";
    println!(
        "{} distinct observed byte bigrams",
        byte_bigrams(text).len()
    );
    println!("Each following byte is one of 256 target class IDs.");
    println!("Complete the predictor, then run cargo test.");
}
#[test]
fn predicts_from_counts() {
    assert_eq!(most_likely_after(b"ababac", b'a'), Some(b'b'));
}
