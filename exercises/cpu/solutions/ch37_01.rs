fn split(document_id: u64) -> &'static str {
    match document_id % 10 {
        0 => "test",
        1 => "validation",
        _ => "train",
    }
}
fn main() {
    println!("Produce whole-document splits; a later consumer builds windows within each one.");
}
#[test]
fn whole_document_split_is_stable() {
    assert_eq!(split(10), "test");
    assert_eq!(split(11), "validation");
    assert_eq!(split(12), "train");
}
