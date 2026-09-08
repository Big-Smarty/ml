fn clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
fn split(_document_id: u64) -> &'static str {
    // TODO: reserve id%10==0 for test, ==1 for validation.
    todo!("guided repair: assign the document split")
}
fn main() {
    let _exercise = split as fn(u64) -> &'static str;
    for text in [" a  small\n document ", "another document"] {
        println!("cleaned: {:?}", clean(text));
    }
    println!("Complete whole-document splits, then run cargo test; windowing comes later.");
}
#[test]
fn whole_document_split_is_stable() {
    assert_eq!(split(20), "test");
    assert_eq!(split(21), "validation");
    assert_eq!(split(22), "train");
}
