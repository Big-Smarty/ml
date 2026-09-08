fn predict(raw: f64, mean: f64, scale: f64, weight: f64, bias: f64) -> f64 {
    weight * ((raw - mean) / scale) + bias
}

#[cfg(test)]
fn parse_request(body: &str) -> Result<f64, &'static str> {
    // TODO: accept exactly schema=1&x=<finite number> and reject extra fields.
    let _ = body;
    todo!("validate the serving boundary")
}

fn main() {
    let score = predict(12.0, 10.0, 2.0, 0.75, -0.25);
    println!("version=v1 schema=1 raw=12.0 score={score:.3}");
    println!("The model and its preprocessing values must travel together.");
}

#[test]
fn request_schema_is_explicit() {
    assert_eq!(parse_request("schema=1&x=12.5"), Ok(12.5));
    assert!(parse_request("x=12.5").is_err());
}
