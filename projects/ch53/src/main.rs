//! A versioned linear model, atomic checkpoint, HTTP boundary, drift monitor, and rollback.
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

const TRAIN: [(f64, f64); 5] = [
    (8.0, -1.0),
    (9.0, 0.0),
    (10.0, 1.0),
    (11.0, 2.0),
    (12.0, 3.0),
];

#[derive(Clone, Debug, PartialEq)]
struct Model {
    format: u32,
    model_version: u32,
    input_schema: u32,
    mean: f64,
    scale: f64,
    weight: f64,
    bias: f64,
    data_hash: u64,
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

    fn encode(&self) -> String {
        format!(
            "MLMODEL {} {} {} {} {} {} {} {}\n",
            self.format,
            self.model_version,
            self.input_schema,
            self.mean,
            self.scale,
            self.weight,
            self.bias,
            self.data_hash
        )
    }

    fn decode(text: &str) -> Result<Self, &'static str> {
        let fields: Vec<_> = text.split_whitespace().collect();
        if fields.len() != 9 || fields[0] != "MLMODEL" {
            return Err("malformed checkpoint");
        }
        let model = Self {
            format: fields[1].parse().map_err(|_| "invalid format version")?,
            model_version: fields[2].parse().map_err(|_| "invalid model version")?,
            input_schema: fields[3].parse().map_err(|_| "invalid schema version")?,
            mean: fields[4].parse().map_err(|_| "invalid mean")?,
            scale: fields[5].parse().map_err(|_| "invalid scale")?,
            weight: fields[6].parse().map_err(|_| "invalid weight")?,
            bias: fields[7].parse().map_err(|_| "invalid bias")?,
            data_hash: fields[8].parse().map_err(|_| "invalid data hash")?,
        };
        if model.format != 1 || model.input_schema != 1 {
            return Err("unsupported checkpoint or input schema");
        }
        if model.scale <= 0.0
            || [model.mean, model.scale, model.weight, model.bias]
                .iter()
                .any(|v| !v.is_finite())
        {
            return Err("checkpoint numbers must be finite and scale positive");
        }
        Ok(model)
    }
}

fn data_hash(data: &[(f64, f64)]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &(x, y) in data {
        for byte in x
            .to_bits()
            .to_le_bytes()
            .into_iter()
            .chain(y.to_bits().to_le_bytes())
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

fn train(version: u32, data: &[(f64, f64)]) -> Result<Model, &'static str> {
    if data.is_empty() || data.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
        return Err("training data must be finite and nonempty");
    }
    let mean = data.iter().map(|(x, _)| x).sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|(x, _)| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let scale = variance.sqrt();
    if !mean.is_finite() || !scale.is_finite() || scale <= 0.0 {
        return Err("training moments must be finite and inputs must vary");
    }
    let mut weight = 0.0;
    let mut bias = 0.0;
    for _ in 0..300 {
        let (dw, db) = data.iter().fold((0.0, 0.0), |(dw, db), &(x, y)| {
            let normalized = (x - mean) / scale;
            let error = weight * normalized + bias - y;
            (dw + 2.0 * error * normalized, db + 2.0 * error)
        });
        weight -= 0.05 * dw / data.len() as f64;
        bias -= 0.05 * db / data.len() as f64;
    }
    if !weight.is_finite() || !bias.is_finite() {
        return Err("training produced nonfinite parameters");
    }
    Ok(Model {
        format: 1,
        model_version: version,
        input_schema: 1,
        mean,
        scale,
        weight,
        bias,
        data_hash: data_hash(data),
    })
}

fn save_atomic(path: &Path, model: &Model) -> Result<(), Box<dyn std::error::Error>> {
    Model::decode(&model.encode())?;
    let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let temporary = path.with_extension(format!("tmp-{}-{id}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        file.write_all(model.encode().as_bytes())?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result?;
    Ok(())
}

fn load(path: &Path) -> Result<Model, Box<dyn std::error::Error>> {
    let mut text = String::new();
    fs::File::open(path)?.take(1025).read_to_string(&mut text)?;
    if text.len() > 1024 {
        return Err("model artifact exceeds 1024 bytes".into());
    }
    Ok(Model::decode(&text)?)
}

#[derive(Default)]
struct Monitor {
    count: usize,
    sum: f64,
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
        Ok(())
    }

    fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.sum / self.count as f64)
    }

    fn drift_from(&self, baseline: f64, threshold: f64) -> Result<bool, &'static str> {
        if !baseline.is_finite() || !threshold.is_finite() || threshold < 0.0 {
            return Err("drift settings must be finite and threshold nonnegative");
        }
        Ok(self
            .mean()
            .is_some_and(|mean| (mean - baseline).abs() > threshold))
    }
}

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

fn parse_body(body: &str) -> Result<f64, &'static str> {
    let mut schema = None;
    let mut x = None;
    for field in body.split('&') {
        let (name, value) = field.split_once('=').ok_or("fields must use name=value")?;
        match name {
            "schema" if schema.is_none() => {
                schema = Some(
                    value
                        .parse::<u32>()
                        .map_err(|_| "schema must be an integer")?,
                )
            }
            "x" if x.is_none() => x = Some(value.parse::<f64>().map_err(|_| "x must be a number")?),
            "schema" | "x" => return Err("duplicate request field"),
            _ => return Err("unknown request field"),
        }
    }
    if schema != Some(1) {
        return Err("unsupported input schema");
    }
    let value = x.ok_or("missing x")?;
    if !value.is_finite() || !(-1000.0..=1000.0).contains(&value) {
        return Err("x must be finite and within [-1000,1000]");
    }
    Ok(value)
}

fn read_bounded_line(
    reader: &mut impl BufRead,
    maximum: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut line = String::new();
    let bytes = reader.take((maximum + 1) as u64).read_line(&mut line)?;
    if bytes == 0 {
        return Err("unexpected end of HTTP headers".into());
    }
    if bytes > maximum || !line.ends_with('\n') {
        return Err("HTTP line is too long".into());
    }
    Ok(line)
}

fn read_request(reader: &mut impl BufRead) -> Result<String, Box<dyn std::error::Error>> {
    let request_line = read_bounded_line(reader, 256)?;
    if request_line.trim_end() != "POST /predict HTTP/1.1" {
        return Err("expected POST /predict HTTP/1.1".into());
    }
    let mut content_length = None;
    let mut header_bytes = request_line.len();
    loop {
        let line = read_bounded_line(reader, 512)?;
        header_bytes += line.len();
        if header_bytes > 2048 {
            return Err("HTTP headers are too large".into());
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        let (name, value) = line.split_once(':').ok_or("malformed HTTP header")?;
        if name.eq_ignore_ascii_case("content-length") {
            if content_length.is_some() {
                return Err("duplicate Content-Length".into());
            }
            content_length = Some(value.trim().parse::<usize>()?);
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            return Err("Transfer-Encoding is unsupported".into());
        }
    }
    let length = content_length.ok_or("missing Content-Length")?;
    if length > 128 {
        return Err("request body is too large".into());
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    Ok(String::from_utf8(body)?)
}

fn handle_stream(
    mut stream: TcpStream,
    model: &Model,
    monitor: &mut Monitor,
) -> Result<(), Box<dyn std::error::Error>> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(5)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let result = read_request(&mut reader)
        .map_err(|_| "invalid HTTP request")
        .and_then(|body| parse_body(&body))
        .and_then(|x| model.predict(x).map(|prediction| (x, prediction)));
    match result {
        Ok((input, prediction)) => {
            monitor.observe(input)?;
            eprintln!(
                "accepted requests={} input_mean={:.4} mean_drift={}",
                monitor.count,
                monitor.mean().ok_or("empty monitor")?,
                monitor.drift_from(model.mean, 1.0)?
            );
            write_response(
                &mut stream,
                200,
                &format!(
                    "version={} prediction={prediction:.6}\n",
                    model.model_version
                ),
            )
        }
        Err(message) => write_response(&mut stream, 400, &format!("error={message}\n")),
    }
}

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    body: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let reason = if status == 200 { "OK" } else { "Bad Request" };
    write!(stream, "HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())?;
    stream.flush()?;
    Ok(())
}

fn serve(address: &str, model: Model) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(address)?;
    println!(
        "serving model v{} on http://{address}/predict",
        model.model_version
    );
    // ponytail: sequential requests cap throughput; use a bounded pool when measurements require it.
    let mut monitor = Monitor::default();
    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                if let Err(error) = handle_stream(stream, &model, &mut monitor) {
                    eprintln!("request failed: {error}");
                }
            }
            Err(error) => eprintln!("accept failed: {error}"),
        }
    }
    Ok(())
}

fn demo() -> Result<(), Box<dyn std::error::Error>> {
    let previous = train(1, &TRAIN)?;
    let mut candidate = train(2, &TRAIN)?;
    candidate.weight *= 1.5;
    let selected = choose_release(&candidate, &previous, &[8.0, 10.0, 12.0], 0.25)?;
    let path = std::env::temp_dir().join(format!("ch53-{}.model", std::process::id()));
    save_atomic(&path, selected)?;
    let restored = load(&path)?;
    fs::remove_file(&path)?;
    let mut monitor = Monitor::default();
    for value in [11.5, 12.0, 12.5, 13.0] {
        monitor.observe(value)?;
    }
    println!(
        "checkpoint: model v{}, schema {}, data hash {:016x}",
        restored.model_version, restored.input_schema, restored.data_hash
    );
    println!(
        "canary rollback selected previous version: {}",
        restored.model_version == previous.model_version
    );
    println!(
        "live input mean {:.2}; drift alert: {}",
        monitor.mean().ok_or("monitor empty")?,
        monitor.drift_from(previous.mean, 1.0)?
    );
    println!(
        "validated prediction for x=12: {:.4}",
        restored.predict(12.0)?
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("train") => {
            let path = Path::new(args.get(2).ok_or("usage: ch53 train MODEL_PATH")?);
            save_atomic(path, &train(1, &TRAIN)?)
        }
        Some("serve") => {
            let address = args.get(2).map_or("127.0.0.1:7878", String::as_str);
            let path = Path::new(args.get(3).ok_or("usage: ch53 serve ADDRESS MODEL_PATH")?);
            serve(address, load(path)?)
        }
        Some(_) => Err("usage: ch53 [train MODEL_PATH | serve ADDRESS MODEL_PATH]".into()),
        None => demo(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_round_trip_and_schema_validation() -> Result<(), Box<dyn std::error::Error>> {
        let model = train(7, &TRAIN)?;
        assert_eq!(Model::decode(&model.encode())?, model);
        assert!(Model::decode("MLMODEL 2 1 1 0 1 1 0 0").is_err());
        assert_eq!(parse_body("schema=1&x=12.5")?, 12.5);
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
        let model = train(1, &TRAIN)?;
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
