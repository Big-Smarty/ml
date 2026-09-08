//! Optional Burn/ndarray CPU parity check for the Chapter 55 scratch oracle.
use burn::{
    backend::{Autodiff, NdArray},
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
};

type Cpu = Autodiff<NdArray<f32>>;

fn values<const D: usize>(tensor: Tensor<Cpu, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().expect("f32 tensor data")
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-6 + 1e-5 * a.abs().max(b.abs()),
            "{a} != {b}"
        );
    }
}

fn run_parity() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();
    let input = Tensor::<Cpu, 2>::from_floats([[1.5, -2.0]], &device);
    let target = Tensor::<Cpu, 2>::from_floats([[1.0, -0.5]], &device);
    // Burn matmul expects [in, out]; the scratch oracle stores [out, in].
    let weight = Tensor::<Cpu, 2>::from_floats([[0.2, 0.7], [-0.4, 0.1]], &device).require_grad();
    let bias = Tensor::<Cpu, 1>::from_floats([0.05, -0.2], &device).require_grad();
    let output = input.clone().matmul(weight.clone()) + bias.clone().unsqueeze_dim::<2>(0);
    let loss = (output.clone() - target).powf_scalar(2.0).mean();
    close(&values(output), &[1.15, 0.65]);
    close(&[loss.clone().into_scalar()], &[0.6725]);

    let gradients = loss.backward();
    let weight_grad = weight.grad(&gradients).ok_or("missing weight gradient")?;
    let bias_grad = bias.grad(&gradients).ok_or("missing bias gradient")?;
    close(
        &values(Tensor::from_inner(weight_grad.clone())),
        &[0.225, 1.725, -0.3, -2.3],
    );
    close(
        &values(Tensor::from_inner(bias_grad.clone())),
        &[0.15, 1.15],
    );

    // Direct tensor SGD keeps the parity boundary visible and avoids optimizer policy.
    let updated = Tensor::<Cpu, 2>::from_inner(weight.inner() - weight_grad.mul_scalar(0.1));
    let updated_bias = Tensor::<Cpu, 1>::from_inner(bias.inner() - bias_grad.mul_scalar(0.1));
    close(&values(updated_bias.clone()), &[0.035, -0.315]);
    let after = input.matmul(updated.clone()) + updated_bias.clone().unsqueeze_dim::<2>(0);
    close(&values(after), &[1.04125, -0.18375]);
    close(&values(updated.clone()), &[0.1775, 0.5275, -0.37, 0.33]);

    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();
    let bytes = recorder.record((updated.inner(), updated_bias.inner()), ())?;
    let (restored_weight, restored_bias): (Tensor<NdArray<f32>, 2>, Tensor<NdArray<f32>, 1>) =
        recorder.load(bytes, &device)?;
    close(
        &restored_weight.to_data().to_vec::<f32>()?,
        &[0.1775, 0.5275, -0.37, 0.33],
    );
    close(&restored_bias.to_data().to_vec::<f32>()?, &[0.035, -0.315]);
    let restored_output = Tensor::<NdArray<f32>, 2>::from_floats([[1.5, -2.0]], &device)
        .matmul(restored_weight)
        + restored_bias.unsqueeze_dim::<2>(0);
    close(
        &restored_output.to_data().to_vec::<f32>()?,
        &[1.04125, -0.18375],
    );
    println!("Burn ndarray CPU forward, loss, gradients, SGD, and record round trip match.");
    Ok(())
}

fn verify_scratch_artifact(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(36)
        .read_to_end(&mut bytes)?;
    if bytes.len() != 35 || &bytes[..11] != b"CH55DENSE01" {
        return Err("invalid scratch model header or length".into());
    }
    let mut p = [0.0f32; 6];
    for (value, bytes) in p.iter_mut().zip(bytes[11..].chunks_exact(4)) {
        *value = f32::from_le_bytes(bytes.try_into()?);
    }
    if p.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite imported model".into());
    }
    let device = Default::default();
    let weight = Tensor::<Cpu, 2>::from_floats([[p[0], p[2]], [p[1], p[3]]], &device);
    let bias = Tensor::<Cpu, 1>::from_floats([p[4], p[5]], &device);
    for x in [[1.5, -2.0], [-1.0, 3.0], [0.0, 0.0]] {
        let expected = [
            p[0] * x[0] + p[1] * x[1] + p[4],
            p[2] * x[0] + p[3] * x[1] + p[5],
        ];
        let output = Tensor::<Cpu, 2>::from_floats([x], &device).matmul(weight.clone())
            + bias.clone().unsqueeze_dim::<2>(0);
        close(&values(output), &expected);
    }
    println!("exported scratch artifact matches Burn on three input probes");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_parity()?;
    let args: Vec<_> = std::env::args().collect();
    match args.as_slice() {
        [_] => Ok(()),
        [_, command, path] if command == "import" => {
            verify_scratch_artifact(std::path::Path::new(path))
        }
        _ => Err("usage: ch55-burn [import SCRATCH_ARTIFACT]".into()),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn optional_binary_executes_parity_assertions() {
        super::run_parity().expect("framework parity");
    }
}
