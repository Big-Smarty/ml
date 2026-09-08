fn parse_schema(field: &str) -> Result<u32, &'static str> {
    // TODO: in this isolated exercise, accept only the canonical field schema=1.
    let _ = field;
    todo!("validate name and version")
}
fn main() {
    println!("This isolated exercise accepts only the canonical schema field.");
}
#[test]
fn schema_version() {
    assert_eq!(parse_schema("schema=1"), Ok(1));
    assert!(parse_schema("schema=2").is_err());
    assert!(parse_schema("schema=01").is_err());
    assert!(parse_schema("schema=+1").is_err());
}
