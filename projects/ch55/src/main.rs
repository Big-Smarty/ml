//! A dependency-free oracle for checking a framework port.
use std::{
    fs,
    io::{Read, Write},
    path::Path,
};

const INPUT: [f32; 2] = [1.5, -2.0];
const TARGET: [f32; 2] = [1.0, -0.5];
const RATE: f32 = 0.1;

#[derive(Clone, Debug, PartialEq)]
struct Dense {
    // Row-major [out_features, in_features].
    weight: [[f32; 2]; 2],
    bias: [f32; 2],
}

impl Dense {
    fn fixture() -> Self {
        Self {
            weight: [[0.2, -0.4], [0.7, 0.1]],
            bias: [0.05, -0.2],
        }
    }

    fn forward(&self, input: [f32; 2]) -> [f32; 2] {
        std::array::from_fn(|o| {
            self.weight[o][0] * input[0] + self.weight[o][1] * input[1] + self.bias[o]
        })
    }

    fn loss_and_grad(&self, input: [f32; 2], target: [f32; 2]) -> (f32, Self) {
        let output = self.forward(input);
        let error: [f32; 2] = std::array::from_fn(|o| output[o] - target[o]);
        let loss = error.iter().map(|x| x * x).sum::<f32>() / error.len() as f32;
        // d(mean(error²))/doutput = error because there are two outputs.
        let grad = Self {
            weight: std::array::from_fn(|o| std::array::from_fn(|i| error[o] * input[i])),
            bias: error,
        };
        (loss, grad)
    }

    fn step(&mut self, grad: &Self, rate: f32) -> Result<(), &'static str> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        if self
            .weight
            .iter()
            .flatten()
            .chain(&self.bias)
            .zip(grad.weight.iter().flatten().chain(&grad.bias))
            .any(|(p, g)| !p.is_finite() || !g.is_finite() || !(p - rate * g).is_finite())
        {
            return Err("nonfinite model, gradient, or update");
        }
        for o in 0..2 {
            for i in 0..2 {
                self.weight[o][i] -= rate * grad.weight[o][i];
            }
            self.bias[o] -= rate * grad.bias[o];
        }
        Ok(())
    }

    fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let mut bytes = b"CH55DENSE01".to_vec();
        for value in self.weight.iter().flatten().chain(&self.bias) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        if self
            .weight
            .iter()
            .flatten()
            .chain(&self.bias)
            .any(|x| !x.is_finite())
        {
            return Err("nonfinite model".into());
        }
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        Ok(())
    }

    fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut bytes = Vec::new();
        fs::File::open(path)?.take(36).read_to_end(&mut bytes)?;
        if bytes.len() != 11 + 6 * 4 || &bytes[..11] != b"CH55DENSE01" {
            return Err("wrong checkpoint header or length".into());
        }
        let mut values = [0.0; 6];
        for (value, chunk) in values.iter_mut().zip(bytes[11..].chunks_exact(4)) {
            *value = f32::from_le_bytes(chunk.try_into()?);
        }
        if values.iter().any(|x| !x.is_finite()) {
            return Err("checkpoint contains a nonfinite value".into());
        }
        Ok(Self {
            weight: [[values[0], values[1]], [values[2], values[3]]],
            bias: [values[4], values[5]],
        })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Dense::fixture();
    let output = model.forward(INPUT);
    let (loss, grad) = model.loss_and_grad(INPUT, TARGET);
    println!("scratch output: [{:.6}, {:.6}]", output[0], output[1]);
    println!("scratch mean squared loss: {loss:.6}");
    println!("scratch weight gradient: {:?}", grad.weight);
    model.step(&grad, RATE)?;
    println!("scratch weights after SGD: {:?}", model.weight);
    let args: Vec<_> = std::env::args().collect();
    let exported = match args.as_slice() {
        [_] => None,
        [_, command, path] if command == "export" => Some(std::path::PathBuf::from(path)),
        _ => return Err("usage: ch55 [export NEW_ARTIFACT_PATH]".into()),
    };
    let path = exported.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!("ch55-dense-{}.bin", std::process::id()))
    });
    model.save(&path)?;
    assert_eq!(model, Dense::load(&path)?);
    if exported.is_none() {
        fs::remove_file(&path)?;
    }
    println!("checkpoint round trip: exact");
    if exported.is_some() {
        println!("exported affine parameters to {}", path.display());
    }
    println!("Run the optional Burn parity program in projects/ch55/framework.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_matches_hand_calculation() {
        let model = Dense::fixture();
        let (loss, grad) = model.loss_and_grad(INPUT, TARGET);
        for (actual, expected) in model.forward(INPUT).into_iter().zip([1.15, 0.65]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert!((loss - 0.6725).abs() < 1e-6);
        for (actual, expected) in grad
            .weight
            .iter()
            .flatten()
            .chain(&grad.bias)
            .zip([0.225, -0.3, 1.725, -2.3, 0.15, 1.15])
        {
            assert!((actual - expected).abs() <= 1e-6 + 1e-5 * expected.abs());
        }
    }

    #[test]
    fn checkpoint_rejects_bad_bytes() {
        let path = std::env::temp_dir().join(format!("ch55-bad-{}.bin", std::process::id()));
        fs::write(&path, b"not a model").unwrap();
        assert!(Dense::load(&path).is_err());
        fs::remove_file(path).unwrap();
    }
}
