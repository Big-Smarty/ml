fn split(document_id: u64) -> &'static str {
    // TODO: reserve remainder 0 for test and 1 for validation.
    let _ = document_id;
    todo!()
}
fn main() {
    println!("Split complete documents before token windows.");
}
#[test]
fn stable_split() {
    assert_eq!(split(10), "test");
    assert_eq!(split(11), "validation");
    assert_eq!(split(12), "train");
}
