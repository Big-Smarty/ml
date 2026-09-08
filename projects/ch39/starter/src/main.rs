use ch36::{Config, Decoder};
use ch38::{TrainConfig, Trainer};
fn next_token_windows(data: &[usize], context: usize) -> Vec<(&[usize], &[usize])> {
    // TODO: return input windows and their one-token-shifted targets.
    let _ = (data, context);
    todo!("guided repair: construct shifted windows")
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _exercise = next_token_windows as fn(&[usize], usize) -> Vec<(&[usize], &[usize])>;
    let data = b"small language model"
        .iter()
        .map(|&b| b as usize)
        .collect::<Vec<_>>();
    let model = Decoder::new(
        Config {
            vocab_size: 256,
            context: 12,
            width: 16,
            heads: 2,
            layers: 1,
            ff_width: 32,
        },
        39,
    )?;
    let trainer = Trainer::new(model, TrainConfig::default(), 3)?;
    println!(
        "Ready to train {} parameters on {} bytes.",
        trainer.model.parameter_count(),
        data.len()
    );
    println!("Complete next_token_windows, then run cargo test.");
    Ok(())
}
#[test]
fn targets_are_shifted_one_token() {
    let d = [1, 2, 3, 4, 5];
    let w = next_token_windows(&d, 2);
    assert_eq!(
        w,
        vec![
            (&d[0..2], &d[1..3]),
            (&d[1..3], &d[2..4]),
            (&d[2..4], &d[3..5])
        ]
    );
}
