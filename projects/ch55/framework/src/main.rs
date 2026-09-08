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
    let weights = Tensor::<Cpu, 2>::from_floats([[0.2, 0.7], [-0.4, 0.1]], &device).require_grad();
    let bias = Tensor::<Cpu, 1>::from_floats([0.05, -0.2], &device).require_grad();
    let output = input.clone().matmul(weights.clone()) + bias.clone().unsqueeze_dim::<2>(0);
    let loss = (output.clone() - target).powf_scalar(2.0).mean();
    close(&values(output), &[1.15, 0.65]);
    close(&[loss.clone().into_scalar()], &[0.6725]);

    let gradients = loss.backward();
    let weights_gradient = weights.grad(&gradients).ok_or("missing weight gradient")?;
    let bias_gradient = bias.grad(&gradients).ok_or("missing bias gradient")?;
    close(
        &values(Tensor::from_inner(weights_gradient.clone())),
        &[0.225, 1.725, -0.3, -2.3],
    );
    close(
        &values(Tensor::from_inner(bias_gradient.clone())),
        &[0.15, 1.15],
    );

    // Direct tensor SGD keeps the parity boundary visible and avoids optimizer policy.
    let updated_weights =
        Tensor::<Cpu, 2>::from_inner(weights.inner() - weights_gradient.mul_scalar(0.1));
    let updated_bias = Tensor::<Cpu, 1>::from_inner(bias.inner() - bias_gradient.mul_scalar(0.1));
    close(&values(updated_bias.clone()), &[0.035, -0.315]);
    let after = input.matmul(updated_weights.clone()) + updated_bias.clone().unsqueeze_dim::<2>(0);
    close(&values(after), &[1.04125, -0.18375]);
    close(
        &values(updated_weights.clone()),
        &[0.1775, 0.5275, -0.37, 0.33],
    );

    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();
    let bytes = recorder.record((updated_weights.inner(), updated_bias.inner()), ())?;
    let (restored_weights, restored_bias): (Tensor<NdArray<f32>, 2>, Tensor<NdArray<f32>, 1>) =
        recorder.load(bytes, &device)?;
    close(
        &restored_weights.to_data().to_vec::<f32>()?,
        &[0.1775, 0.5275, -0.37, 0.33],
    );
    close(&restored_bias.to_data().to_vec::<f32>()?, &[0.035, -0.315]);
    let restored_output = Tensor::<NdArray<f32>, 2>::from_floats([[1.5, -2.0]], &device)
        .matmul(restored_weights)
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
    let mut parameters = [0.0f32; 6];
    for (value, bytes) in parameters.iter_mut().zip(bytes[11..].chunks_exact(4)) {
        *value = f32::from_le_bytes(bytes.try_into()?);
    }
    if parameters.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite imported model".into());
    }
    let device = Default::default();
    let weights = Tensor::<Cpu, 2>::from_floats(
        [
            [parameters[0], parameters[2]],
            [parameters[1], parameters[3]],
        ],
        &device,
    );
    let bias = Tensor::<Cpu, 1>::from_floats([parameters[4], parameters[5]], &device);
    for input in [[1.5, -2.0], [-1.0, 3.0], [0.0, 0.0]] {
        let expected = [
            parameters[0] * input[0] + parameters[1] * input[1] + parameters[4],
            parameters[2] * input[0] + parameters[3] * input[1] + parameters[5],
        ];
        let output = Tensor::<Cpu, 2>::from_floats([input], &device).matmul(weights.clone())
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
