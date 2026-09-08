struct Model {
    model_version: u32,
    input_schema: u32,
    mean: f64,
    scale: f64,
    weight: f64,
    bias: f64,
}

impl Model {
    fn predict(&self, raw: f64) -> Result<f64, &'static str> {
        if self.scale <= 0.0
            || [self.mean, self.scale, self.weight, self.bias]
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err("model parameters are invalid");
        }
        if !raw.is_finite() || !(-1000.0..=1000.0).contains(&raw) {
            return Err("x must be finite and within [-1000,1000]");
        }
        let prediction = self.weight * ((raw - self.mean) / self.scale) + self.bias;
        prediction
            .is_finite()
            .then_some(prediction)
            .ok_or("prediction is nonfinite")
    }
}

#[cfg(test)]
fn parse_body(body: &str) -> Result<f64, &'static str> {
    // TODO: accept one supported integer schema field and one finite x field; reject extras.
    let _ = body;
    todo!("validate the serving boundary")
}

fn main() -> Result<(), &'static str> {
    let model = Model {
        model_version: 1,
        input_schema: 1,
        mean: 10.0,
        scale: 2.0,
        weight: 0.75,
        bias: -0.25,
    };
    let prediction = model.predict(12.0)?;
    println!(
        "version=v{} schema={} raw=12.0 prediction={prediction:.3}",
        model.model_version, model.input_schema
    );
    println!("The model and its preprocessing values must travel together.");
    Ok(())
}

#[test]
fn request_schema_is_explicit() {
    assert_eq!(parse_body("schema=1&x=12.5"), Ok(12.5));
    assert_eq!(parse_body("schema=01&x=12.5"), Ok(12.5));
    assert_eq!(parse_body("schema=+1&x=12.5"), Ok(12.5));
    assert!(parse_body("x=12.5").is_err());
}

#[test]
fn model_uses_stored_preprocessing() -> Result<(), &'static str> {
    let model = Model {
        model_version: 1,
        input_schema: 1,
        mean: 10.0,
        scale: 2.0,
        weight: 0.75,
        bias: -0.25,
    };
    assert_eq!(model.predict(12.0)?, 0.5);
    assert!(model.predict(f64::NAN).is_err());
    Ok(())
}
