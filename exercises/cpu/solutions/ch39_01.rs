fn shifted(data: &[u8]) -> (&[u8], &[u8]) {
    (&data[..data.len() - 1], &data[1..])
}
fn main() {
    println!("Every byte predicts the byte one position ahead.");
}
#[test]
fn one_token_shift() {
    assert_eq!(shifted(b"rust"), (&b"rus"[..], &b"ust"[..]));
}
