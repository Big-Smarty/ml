use ch36::{Config, Decoder};
use ch38::{TrainConfig, Trainer};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = b"rust learns. rust predicts. rust learns."
        .iter()
        .map(|&b| b as usize)
        .collect::<Vec<_>>();
    let model = Decoder::new(
        Config {
            vocab_size: 256,
            context: 8,
            width: 16,
            heads: 2,
            layers: 1,
            ff_width: 32,
        },
        38,
    )?;
    let mut trainer = Trainer::new(model, TrainConfig::default(), 99)?;
    let first = trainer.train_accumulated(&data, 8, 2)?;
    for _ in 0..24 {
        trainer.train_accumulated(&data, 8, 2)?;
    }
    let last = trainer.train_accumulated(&data, 8, 2)?;
    let path = std::env::temp_dir().join("ch38-resume.bin");
    trainer.save(&path)?;
    let restored = Trainer::load(&path)?;
    println!(
        "accumulated-loss {first:.3} -> {last:.3}; step {}",
        trainer.step
    );
    println!("model + Adam moments use {} bytes", trainer.memory_bytes());
    println!(
        "restored step {}, cursor {}, RNG {}",
        restored.step,
        restored.data_cursor,
        restored.rng.state()
    );
    Ok(())
}
