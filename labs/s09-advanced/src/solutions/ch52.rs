//! Worked solution: Normalize each modality embedding and score every image/caption pair. Compute a stable cross-entropy separately for each row and column and average over both directions. This penalizes mismatches as well as attracting pairs; retrieval still depends on the supplied candidate set and its labels.
//! Compare intermediate quantities with the lesson before using the final goal check.
//! Worked solution for chapter 52. Read the comments and lesson explanations before comparing.
include!("../common/ch52.rs");
include!("../checks/ch52.rs");
fn normalize(vector: [f64; 2]) -> Result<[f64; 2], &'static str> {
    let norm = (vector[0].powi(2) + vector[1].powi(2)).sqrt();
    if !norm.is_finite() || norm < 1e-12 {
        return Err("embedding norm is zero or nonfinite");
    }
    Ok([vector[0] / norm, vector[1] / norm])
}

impl DualEncoder {
    fn similarities(
        &self,
        images: &[[f64; 4]; 3],
        captions: &[&str; 3],
    ) -> Result<[[f64; 3]; 3], &'static str> {
        let mut similarities = [[0.0; 3]; 3];
        for (i, image) in images.iter().copied().enumerate() {
            let ie = self.image_embedding(image)?;
            for (j, caption) in captions.iter().copied().enumerate() {
                let te = self.caption_embedding(caption)?;
                similarities[i][j] = ie[0] * te[0] + ie[1] * te[1];
            }
        }
        Ok(similarities)
    }
}

impl DualEncoder {
    fn loss(
        &self,
        images: &[[f64; 4]; 3],
        captions: &[&str; 3],
        temperature: f64,
    ) -> Result<f64, &'static str> {
        let logits = self.logits(images, captions, temperature)?;
        let mut total = 0.0;
        for (i, row) in logits.iter().copied().enumerate() {
            total += cross_entropy_from_logits(row, i);
            total += cross_entropy_from_logits([logits[0][i], logits[1][i], logits[2][i]], i);
        }
        let loss = total / 6.0;
        if !loss.is_finite() {
            return Err("contrastive loss is nonfinite");
        }
        Ok(loss)
    }
}

impl DualEncoder {
    fn recall_at_one(
        &self,
        images: &[[f64; 4]; 3],
        candidate_captions: &[&str; 3],
    ) -> Result<f64, &'static str> {
        let mut correct = 0;
        for (expected, image) in images.iter().copied().enumerate() {
            let ie = self.image_embedding(image)?;
            let best = candidate_captions
                .iter()
                .copied()
                .enumerate()
                .map(|(index, caption)| {
                    let te = self.caption_embedding(caption)?;
                    Ok((index, ie[0] * te[0] + ie[1] * te[1]))
                })
                .collect::<Result<Vec<_>, &'static str>>()?
                .into_iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .ok_or("caption set must not be empty")?
                .0;
            correct += usize::from(best == expected);
        }
        Ok(correct as f64 / images.len() as f64)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_improves_loss_and_retrieval() -> Result<(), &'static str> {
        let initial = initial_model();
        let model = initial.train(&IMAGES, &NAMES, 300, 0.03, TEMPERATURE)?;
        assert!(
            model.loss(&IMAGES, &NAMES, TEMPERATURE)?
                < initial.loss(&IMAGES, &NAMES, TEMPERATURE)? * 0.25
        );
        assert_ne!(model.image_weights, initial.image_weights);
        assert_ne!(model.text_weights, initial.text_weights);
        assert_eq!(model.recall_at_one(&IMAGES, &NAMES)?, 1.0);
        Ok(())
    }

    #[test]
    fn supplied_data_candidates_and_temperature_are_used() -> Result<(), &'static str> {
        let model = initial_model();
        let mut alternate_images = IMAGES;
        alternate_images[0] = [0.0, 1.0, 0.0, 1.0];
        assert_ne!(
            model.loss(&alternate_images, &NAMES, TEMPERATURE)?,
            model.loss(&IMAGES, &NAMES, TEMPERATURE)?
        );

        let logits = model.logits(&IMAGES, &NAMES, TEMPERATURE)?;
        let warmer_logits = model.logits(&IMAGES, &NAMES, 2.0 * TEMPERATURE)?;
        assert!((warmer_logits[0][0] - logits[0][0] / 2.0).abs() < 1e-12);

        let tied_candidates = [NAMES[0], NAMES[0], NAMES[0]];
        assert_eq!(model.recall_at_one(&IMAGES, &tied_candidates)?, 1.0 / 3.0);
        Ok(())
    }

    #[test]
    fn invalid_embeddings_are_rejected() {
        let zero = DualEncoder {
            image_weights: [[0.0; 4]; 2],
            text_weights: [[0.0; 3]; 2],
        };
        assert!(zero.image_embedding(IMAGES[0]).is_err());
        assert!(initial_model().image_embedding([f64::NAN; 4]).is_err());
        assert_eq!(caption_features("two vertical pixels"), Ok([1.0, 0.0, 0.0]));
        assert!(caption_features("unknown description").is_err());
        assert!(initial_model().logits(&IMAGES, &NAMES, 0.0).is_err());
        assert!(initial_model()
            .logits(&IMAGES, &NAMES, f64::MIN_POSITIVE / 1024.0)
            .is_err());
        let extreme = DualEncoder {
            image_weights: [[1.0; 4], [0.0; 4]],
            text_weights: [[1.0, -1.0, 1.0], [0.0; 3]],
        };
        assert!(extreme.loss(&IMAGES, &NAMES, 1e-308).is_err());
    }
}
