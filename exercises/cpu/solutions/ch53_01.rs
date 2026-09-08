fn parse_schema(field: &str) -> Result<u32, &'static str> {
    match field {
        "schema=1" => Ok(1),
        _ => Err("unsupported schema"),
    }
}
fn main() {
    println!("A serving boundary rejects unknown schemas.");
}
#[test]
fn schema_version() {
    assert_eq!(parse_schema("schema=1"), Ok(1));
    assert!(parse_schema("schema=2").is_err());
}
