fn shifted(data: &[u8]) -> (&[u8], &[u8]) {
    // TODO: return all but last as input and all but first as targets.
    let _ = data;
    todo!()
}
fn main() {
    println!("Every byte predicts the byte one position ahead.");
}
#[test]
fn one_token_shift() {
    assert_eq!(shifted(b"rust"), (&b"rus"[..], &b"ust"[..]));
}
