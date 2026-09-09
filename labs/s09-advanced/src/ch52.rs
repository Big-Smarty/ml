//! Chapter 52 learner algorithms. Baseline attracts paired embeddings without negatives. Implement normalized full pair scoring, symmetric row/column cross-entropy, and retrieval under explicit candidates.
include!("common/ch52.rs");
include!("checks/ch52.rs");
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
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err("temperature must be finite and positive");
        }
        // Pair-attraction baseline: no competition with other images or captions.
        // Replace with the full similarity matrix and both contrastive directions.
        let mut sum = 0.0;
        for (&image, &caption) in images.iter().zip(captions) {
            let a = self.image_embedding(image)?;
            let b = self.caption_embedding(caption)?;
            sum += a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
        }
        let loss = sum / images.len() as f64;
        loss.is_finite()
            .then_some(loss)
            .ok_or("alignment loss became nonfinite")
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
