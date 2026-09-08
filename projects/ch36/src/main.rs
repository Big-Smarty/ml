use ch36::{Config, Decoder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        vocab_size: 8,
        context: 4,
        width: 8,
        heads: 2,
        layers: 1,
        ff_width: 16,
    };
    let mut model = Decoder::new(config, 36)?;
    let input = [1, 2, 3, 1];
    let targets = [2, 3, 1, 2];
    let learning_rate = 0.08;
    let initial = model.loss(&input, &targets)?;
    for _ in 0..60 {
        let gradient = model.loss_and_gradient(&input, &targets)?;
        model.apply_sgd(&gradient, learning_rate)?;
    }
    let final_loss = model.loss(&input, &targets)?;
    println!("{} trainable parameters", model.parameter_count());
    println!("loss: {initial:.4} -> {final_loss:.4}");
    println!("Representative embeddings, attention, norms, feed-forward, and output weights were updated.");
    Ok(())
}
