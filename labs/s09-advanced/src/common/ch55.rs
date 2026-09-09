use std::io::{Read, Write};
use std::path::Path;

// A fixed non-square affine interchange fixture: [out=2,in=3], then two biases.
#[derive(Debug, Clone, PartialEq)]
struct Dense {
    weights: [[f64; 3]; 2],
    bias: [f64; 2],
}
#[derive(Debug, Clone)]
struct Runtime {
    weights: [[f64; 2]; 3],
    bias: [f64; 2],
}
const INPUT: [f64; 3] = [1.5, -2.0, 0.5];
const TARGET: [f64; 2] = [1.0, -0.5];
impl Dense {
    fn fixture() -> Self {
        Self {
            weights: [[0.2, -0.4, 0.6], [0.7, 0.1, -0.2]],
            bias: [0.05, -0.2],
        }
    }
    fn forward(&self, x: [f64; 3]) -> [f64; 2] {
        std::array::from_fn(|o| {
            self.weights[o]
                .iter()
                .zip(x)
                .map(|(w, x)| w * x)
                .sum::<f64>()
                + self.bias[o]
        })
    }
    fn gradient(&self, x: [f64; 3], target: [f64; 2]) -> Self {
        let output = self.forward(x);
        let error = std::array::from_fn::<_, 2, _>(|o| output[o] - target[o]);
        Self {
            weights: std::array::from_fn(|o| x.map(|x| error[o] * x)),
            bias: error,
        }
    }
    fn encode(&self) -> Result<Vec<u8>, String> {
        if self
            .weights
            .iter()
            .flatten()
            .chain(&self.bias)
            .any(|x| !x.is_finite())
        {
            return Err("nonfinite artifact parameter".into());
        }
        let mut bytes = b"S09AFF01".to_vec();
        for v in self.weights.iter().flatten().chain(&self.bias) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Ok(bytes)
    }
    fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 72 || &bytes[..8] != b"S09AFF01" {
            return Err("expected S09AFF01 and eight little-endian f64 values (72 bytes)".into());
        }
        let mut values = [0.0; 8];
        for (v, chunk) in values.iter_mut().zip(bytes[8..].chunks_exact(8)) {
            *v = f64::from_le_bytes(chunk.try_into().map_err(|_| "bad float length")?);
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err("artifact contains nonfinite values".into());
        }
        Ok(Self {
            weights: [
                [values[0], values[1], values[2]],
                [values[3], values[4], values[5]],
            ],
            bias: [values[6], values[7]],
        })
    }
}
fn imported(path: &Path) -> Result<Dense, String> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take(73)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    Dense::decode(&bytes)
}
pub fn run(args: &[String]) -> Result<(), String> {
    let model = match args {
        [] => Dense::fixture(),
        [command, path] if command == "import" => imported(Path::new(path))?,
        [command, path] if command == "export" => {
            let m = Dense::fixture();
            let bytes = m.encode()?;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|e| e.to_string())?;
            file.write_all(&bytes)
                .and_then(|()| file.sync_all())
                .map_err(|e| e.to_string())?;
            println!("exported {} bytes to {path}", bytes.len());
            m
        }
        [flag] if flag == "--framework" => {
            let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../projects/ch55/framework/Cargo.toml");
            let status = std::process::Command::new("cargo")
                .args(["run", "--manifest-path"])
                .arg(manifest)
                .status()
                .map_err(|e| e.to_string())?;
            return if status.success() {
                Ok(())
            } else {
                Err("optional pinned Burn reference failed".into())
            };
        }
        _ => return Err("usage: 55 [export NEW_PATH | import PATH | --framework]".into()),
    };
    let restored = Dense::decode(&model.encode()?)?;
    let runtime = Runtime {
        weights: map_weights(&restored.weights),
        bias: restored.bias,
    };
    println!(
        "input {INPUT:?}; [out,in] output {:?}; [in,out] runtime {:?}",
        model.forward(INPUT),
        runtime.forward(INPUT)
    );
    println!("forward/gradient/update parity: {:?}", compare(&restored));
    println!("Custom fixed-shape format, not ONNX. --framework explicitly runs the separate pinned Burn 2x2 reference and may download packages.");
    Ok(())
}
pub fn check() -> Result<(), String> {
    let model = Dense::fixture();
    let runtime = Runtime {
        weights: map_weights(&model.weights),
        bias: model.bias,
    };
    for x in [INPUT, [-1.0, 3.0, 0.25], [0.0; 3]] {
        crate::ensure(
            runtime
                .forward(x)
                .iter()
                .zip(model.forward(x))
                .all(|(&a, b)| crate::close(a, b)),
            "runtime forward coordinates differ from scratch",
        )?;
    }
    let expected = model.gradient(INPUT, TARGET);
    let gradient = runtime.gradient(INPUT, TARGET);
    for i in 0..3 {
        for o in 0..2 {
            crate::ensure(
                crate::close(gradient.weights[i][o], expected.weights[o][i]),
                "runtime gradient must preserve input/output coordinates",
            )?;
        }
    }
    crate::ensure(
        gradient
            .bias
            .iter()
            .zip(expected.bias)
            .all(|(&a, b)| crate::close(a, b)),
        "runtime bias gradient differs",
    )?;
    let mut updated = runtime.clone();
    updated.update(&gradient, 0.1);
    crate::ensure(
        updated
            .weights
            .iter()
            .flatten()
            .zip(
                runtime
                    .weights
                    .iter()
                    .flatten()
                    .zip(gradient.weights.iter().flatten()),
            )
            .all(|(&after, (&before, &g))| crate::close(after, before - 0.1 * g)),
        "runtime update did not apply the gradient",
    )?;
    crate::ensure(
        updated
            .bias
            .iter()
            .zip(runtime.bias.iter().zip(gradient.bias))
            .all(|(&after, (&before, g))| crate::close(after, before - 0.1 * g)),
        "runtime bias update differs",
    )?;
    compare(&model)?;
    crate::ensure(
        verify_parity(&[1.0, 2.0], &[2.0, 1.0]).is_err(),
        "coordinate errors must not cancel",
    )?;
    crate::ensure(
        verify_parity(&[f64::INFINITY], &[f64::INFINITY]).is_err(),
        "nonfinite parity cannot pass",
    )?;
    let bytes = model.encode()?;
    crate::ensure(
        Dense::decode(&bytes)? == model,
        "round-trip changed exact parameters",
    )?;
    crate::ensure(
        Dense::decode(&bytes[..71]).is_err(),
        "truncated artifact accepted",
    )?;
    let mut bad = bytes;
    bad[8..16].copy_from_slice(&f64::NAN.to_le_bytes());
    crate::ensure(Dense::decode(&bad).is_err(), "NaN artifact accepted")
}
