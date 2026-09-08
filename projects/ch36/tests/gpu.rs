#![cfg(feature = "gpu")]

use ch36::{Config, Decoder};

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && e.is_finite());
        assert!(
            (a - e).abs() <= 1e-4 + 1e-3 * e.abs(),
            "coordinate {index}: GPU {a}, CPU {e}"
        );
    }
}

#[test]
#[ignore = "requires a hardware Vulkan GPU; run with --features gpu -- --ignored"]
fn gpu_forward_backward_and_updates_match_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let device = ch30::Gpu::new()?;
    let mut cpu = Decoder::new(
        Config {
            vocab_size: 11,
            context: 3,
            width: 6,
            heads: 2,
            layers: 2,
            ff_width: 7,
        },
        360,
    )?;
    let mut gpu = cpu.clone();
    let input = [1, 4, 1];
    let targets = [4, 1, 8];
    for _ in 0..3 {
        close(
            &gpu.forward_with_gpu(&input, &device)?,
            &cpu.forward(&input)?,
        );
        let oracle = cpu.loss_and_gradient(&input, &targets)?;
        let actual = gpu.loss_and_gradient_with_gpu(&input, &targets, &device)?;
        close(&[actual.loss], &[oracle.loss]);
        close(&actual.values, &oracle.values);
        cpu.apply_sgd(&oracle, 0.02)?;
        gpu.apply_sgd(&actual, 0.02)?;
        close(gpu.parameters(), cpu.parameters());
    }
    println!(
        "Two-layer odd-shape decoder: GPU logits, every gradient, and three updates match CPU."
    );
    Ok(())
}
