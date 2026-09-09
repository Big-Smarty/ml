//! Worked solution: A candidate promotes only when every declared raw-input canary agrees within tolerance. Monitor the most recent four accepted inputs by subtracting evicted values from the running sum; retain the lifetime count separately. Artifact and HTTP validation remain supplied boundaries.
//! Compare intermediate quantities with the lesson before using the final goal check.
//! Worked solution for chapter 53. Read the comments and lesson explanations before comparing.
include!("../common/ch53.rs");
include!("../checks/ch53.rs");
fn choose_release<'a>(
    candidate: &'a Model,
    previous: &'a Model,
    canary: &[f64],
    tolerance: f64,
) -> Result<&'a Model, &'static str> {
    if canary.is_empty() || !tolerance.is_finite() || tolerance < 0.0 {
        return Err("canary inputs and tolerance are required");
    }
    for &input in canary {
        if (candidate.predict(input)? - previous.predict(input)?).abs() > tolerance {
            return Ok(previous);
        }
    }
    Ok(candidate)
}

impl Monitor {
    fn observe(&mut self, value: f64) -> Result<(), &'static str> {
        if !value.is_finite() {
            return Err("monitor values must be finite");
        }
        let count = self.count.checked_add(1).ok_or("monitor count overflow")?;
        let sum = self.sum + value;
        if !sum.is_finite() {
            return Err("monitor sum overflow");
        }
        self.count = count;
        self.sum = sum;
        self.window.push_back(value);
        if self.window.len() > 4 {
            self.sum -= self.window.pop_front().ok_or("missing oldest value")?;
        }
        Ok(())
    }
}

impl Monitor {
    fn drift_from(&self, baseline: f64, threshold: f64) -> Result<bool, &'static str> {
        if !baseline.is_finite() || !threshold.is_finite() || threshold < 0.0 {
            return Err("drift settings must be finite and threshold nonnegative");
        }
        Ok(self
            .mean()
            .is_some_and(|mean| (mean - baseline).abs() > threshold))
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_round_trip_and_schema_validation() -> Result<(), Box<dyn std::error::Error>> {
        let model = train(&TRAIN, 7)?;
        assert_eq!(Model::decode(&model.encode())?, model);
        assert!(Model::decode("MLMODEL 2 1 1 0 1 1 0 0").is_err());
        let fixture = Model {
            format: 1,
            model_version: 1,
            input_schema: 1,
            mean: 10.0,
            scale: 2.0,
            weight: 0.75,
            bias: -0.25,
            data_hash: 0,
        };
        assert_eq!(fixture.predict(12.0)?, 0.5);
        assert_eq!(parse_body("schema=1&x=12.5")?, 12.5);
        assert_eq!(parse_body("schema=01&x=12.5")?, 12.5);
        assert_eq!(parse_body("schema=+1&x=12.5")?, 12.5);
        assert!(parse_body("schema=1&x=NaN").is_err());
        assert!(parse_body("schema=1&x=2&extra=3").is_err());
        let mut valid = std::io::Cursor::new(
            b"POST /predict HTTP/1.1\r\nContent-Length: 15\r\n\r\nschema=1&x=12.0",
        );
        assert_eq!(read_request(&mut valid)?, "schema=1&x=12.0");
        let mut ambiguous = std::io::Cursor::new(
            b"POST /predict HTTP/1.1\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n",
        );
        assert!(read_request(&mut ambiguous).is_err());
        let mut chunked =
            std::io::Cursor::new(b"POST /predict HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n");
        assert!(read_request(&mut chunked).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires host loopback networking"]
    fn serving_monitor_and_rollback_are_real() -> Result<(), Box<dyn std::error::Error>> {
        let model = train(&TRAIN, 1)?;
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server_model = model.clone();
        let server = std::thread::spawn(move || -> Result<(), String> {
            let (stream, _) = listener.accept().map_err(|e| e.to_string())?;
            handle_stream(stream, &server_model, &mut Monitor::default()).map_err(|e| e.to_string())
        });
        let mut client = TcpStream::connect(address)?;
        client.write_all(b"POST /predict HTTP/1.1\r\nContent-Length: 15\r\n\r\nschema=1&x=12.0")?;
        let mut response = String::new();
        client.read_to_string(&mut response)?;
        assert!(response.starts_with("HTTP/1.1 200"));
        server
            .join()
            .map_err(|_| "server thread panicked")?
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        let mut monitor = Monitor::default();
        for value in [14.0, 15.0] {
            monitor.observe(value)?;
        }
        assert!(monitor.drift_from(10.0, 1.0)?);
        let mut bad = model.clone();
        bad.weight *= 2.0;
        assert_eq!(
            choose_release(&bad, &model, &[8.0, 12.0], 0.1)?.model_version,
            1
        );
        Ok(())
    }
}
