use ch36::{Config, Decoder, ParameterSpan};
use std::{collections::HashMap, error::Error, ops::Range};

#[derive(Default)]
struct LayerKv {
    keys: Vec<f32>,
    values: Vec<f32>,
}

struct CachedDecoder<'a> {
    model: &'a Decoder,
    spans: HashMap<String, Range<usize>>,
    layers: Vec<LayerKv>,
    next_position: usize,
}

impl<'a> CachedDecoder<'a> {
    fn new(model: &'a Decoder) -> Self {
        let spans = model
            .parameter_spans()
            .into_iter()
            .map(|s: ParameterSpan| (s.name, s.start..s.end))
            .collect();
        let layers = (0..model.config().layers)
            .map(|_| LayerKv::default())
            .collect();
        Self {
            model,
            spans,
            layers,
            next_position: 0,
        }
    }
    fn parameter_slice(&self, name: &str) -> &[f32] {
        &self.model.parameters()[self.spans[name].clone()]
    }
    /// Cache one token at the next sequence position and return its [vocabulary] logits.
    /// This decoding step advances no optimizer state.
    fn step(&mut self, token_id: usize) -> Result<Vec<f32>, Box<dyn Error>> {
        let c = self.model.config();
        if token_id >= c.vocab_size {
            return Err("token exceeds vocabulary".into());
        }
        if self.next_position >= c.context {
            return Err("cache reached model context; reset and replay a cropped window".into());
        }
        let _token_embedding = self.parameter_slice("token_embedding");
        // TODO: project and append each layer's K/V, compute the new logits,
        // then increment next_position. Reuse cached K/V for earlier positions.
        todo!("implement incremental decoding")
    }
    fn cached_values(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.keys.len() + layer.values.len())
            .sum()
    }
}

fn tiny_model() -> Result<Decoder, ch36::ModelError> {
    Decoder::new(
        Config {
            vocab_size: 12,
            context: 12,
            width: 8,
            heads: 2,
            layers: 2,
            ff_width: 16,
        },
        40,
    )
}
fn main() -> Result<(), Box<dyn Error>> {
    let model = tiny_model()?;
    let cached = CachedDecoder::new(&model);
    println!("new cache stores {} K/V scalars", cached.cached_values());
    let _step = CachedDecoder::step;
    let logits = model.forward(&[1, 2, 3])?;
    let vocab_size = model.config().vocab_size;
    println!(
        "checkpoint: full decoder produced {} logits; last row starts {:.5}",
        logits.len(),
        logits[logits.len() - vocab_size]
    );
    println!("Complete CachedDecoder::step, then test every prefix.");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_matches_every_full_prefix() {
        let model = tiny_model().unwrap();
        let token_ids = [1, 2, 3];
        let mut cached = CachedDecoder::new(&model);
        let vocab_size = model.config().vocab_size;
        for end in 1..=token_ids.len() {
            let cached_logits = cached.step(token_ids[end - 1]).unwrap();
            let full_logits = model.forward(&token_ids[..end]).unwrap();
            let expected = &full_logits[full_logits.len() - vocab_size..];
            assert_eq!(cached_logits.len(), vocab_size, "prefix {end}");
            assert!(
                cached_logits
                    .iter()
                    .zip(expected)
                    .all(|(cached, full)| (cached - full).abs() < 2e-5),
                "prefix {end}"
            );
        }
    }
}
