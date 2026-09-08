fn parse_schema(field: &str) -> Result<u32, &'static str> {
    match field {
        "schema=1" => Ok(1),
        _ => Err("unsupported schema"),
    }
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
