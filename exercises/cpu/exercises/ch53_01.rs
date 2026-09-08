fn parse_schema(field: &str) -> Result<u32, &'static str> {
    // TODO: accept only the exact supported field schema=1.
    let _ = field;
    todo!("validate name and version")
}
fn main() {
    println!("A serving boundary rejects unknown schemas.");
}
#[test]
fn schema_version() {
    assert_eq!(parse_schema("schema=1"), Ok(1));
    assert!(parse_schema("schema=2").is_err());
}
