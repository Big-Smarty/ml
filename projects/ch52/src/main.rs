//! A trainable dual encoder for three course-authored image-caption pairs.

const IMAGES: [[f64; 4]; 3] = [
    [1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 1.0],
];
const NAMES: [&str; 3] = [
    "two vertical pixels",
    "two horizontal pixels",
    "main diagonal",
];
const TEMPERATURE: f64 = 0.2;

#[derive(Clone, Copy, Debug)]
struct DualEncoder {
    image: [[f64; 4]; 2],
    text: [[f64; 3]; 2],
}

fn normalize(vector: [f64; 2]) -> Result<[f64; 2], &'static str> {
    let norm = (vector[0].powi(2) + vector[1].powi(2)).sqrt();
    if !norm.is_finite() || norm < 1e-12 {
        return Err("embedding norm is zero or nonfinite");
    }
    Ok([vector[0] / norm, vector[1] / norm])
}

fn caption_features(caption: &str) -> Result<[f64; 3], &'static str> {
    if caption.is_empty() || !caption.is_ascii() {
        return Err("caption must be nonempty ASCII text");
    }
    let mut features = [0.0; 3];
    for word in caption.split_ascii_whitespace() {
        match word.trim_matches(|character: char| !character.is_ascii_alphabetic()) {
            "vertical" => features[0] += 1.0,
            "horizontal" => features[1] += 1.0,
            "diagonal" => features[2] += 1.0,
            _ => {}
        }
    }
    if features == [0.0; 3] {
        return Err("caption contains none of the three vocabulary words");
    }
    Ok(features)
}

impl DualEncoder {
    fn image_embedding(self, pixels: [f64; 4]) -> Result<[f64; 2], &'static str> {
        if pixels.iter().any(|v| !v.is_finite()) {
            return Err("image pixels must be finite");
        }
        normalize(
            self.image
                .map(|row| row.iter().zip(pixels).map(|(w, x)| w * x).sum()),
        )
    }

    fn text_embedding(self, words: [f64; 3]) -> Result<[f64; 2], &'static str> {
        if words.iter().any(|v| !v.is_finite()) {
            return Err("text features must be finite");
        }
        normalize(
            self.text
                .map(|row| row.iter().zip(words).map(|(w, x)| w * x).sum()),
        )
    }

    fn caption_embedding(self, caption: &str) -> Result<[f64; 2], &'static str> {
        self.text_embedding(caption_features(caption)?)
    }

    fn scores(self) -> Result<[[f64; 3]; 3], &'static str> {
        let mut result = [[0.0; 3]; 3];
        for (i, image) in IMAGES.into_iter().enumerate() {
            let ie = self.image_embedding(image)?;
            for (j, caption) in NAMES.into_iter().enumerate() {
                let te = self.caption_embedding(caption)?;
                result[i][j] = (ie[0] * te[0] + ie[1] * te[1]) / TEMPERATURE;
            }
        }
        Ok(result)
    }

    fn loss(self) -> Result<f64, &'static str> {
        let scores = self.scores()?;
        let mut total = 0.0;
        for (i, row) in scores.iter().copied().enumerate() {
            total += cross_entropy(row, i);
            total += cross_entropy([scores[0][i], scores[1][i], scores[2][i]], i);
        }
        Ok(total / 6.0)
    }

    fn parameters(self) -> [f64; 14] {
        let mut p = [0.0; 14];
        for (slot, value) in p.iter_mut().zip(
            self.image
                .into_iter()
                .flatten()
                .chain(self.text.into_iter().flatten()),
        ) {
            *slot = value;
        }
        p
    }

    fn from_parameters(p: [f64; 14]) -> Self {
        Self {
            image: [[p[0], p[1], p[2], p[3]], [p[4], p[5], p[6], p[7]]],
            text: [[p[8], p[9], p[10]], [p[11], p[12], p[13]]],
        }
    }

    fn train(self, steps: usize, rate: f64) -> Result<Self, &'static str> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        let mut p = self.parameters();
        for _ in 0..steps {
            let mut gradient = [0.0; 14];
            for i in 0..p.len() {
                let mut plus = p;
                let mut minus = p;
                plus[i] += 1e-5;
                minus[i] -= 1e-5;
                gradient[i] = (Self::from_parameters(plus).loss()?
                    - Self::from_parameters(minus).loss()?)
                    / 2e-5;
            }
            for (parameter, slope) in p.iter_mut().zip(gradient) {
                *parameter -= rate * slope;
            }
        }
        let model = Self::from_parameters(p);
        model.loss()?;
        Ok(model)
    }

    fn recall_at_one(self, images: &[[f64; 4]; 3]) -> Result<f64, &'static str> {
        let mut correct = 0;
        for (expected, image) in images.iter().copied().enumerate() {
            let ie = self.image_embedding(image)?;
            let best = NAMES
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

fn cross_entropy(logits: [f64; 3], target: usize) -> f64 {
    let maximum = logits.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let log_sum_exp = maximum
        + logits
            .into_iter()
            .map(|value| (value - maximum).exp())
            .sum::<f64>()
            .ln();
    log_sum_exp - logits[target]
}

fn initial_model() -> DualEncoder {
    DualEncoder {
        image: [[0.30, -0.20, 0.10, 0.25], [-0.15, 0.35, 0.20, -0.10]],
        text: [[0.20, -0.30, 0.10], [0.25, 0.15, -0.20]],
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let initial = initial_model();
    let model = initial.train(300, 0.03)?;
    let evaluation = [
        [0.9, 0.1, 1.0, 0.0],
        [1.0, 0.9, 0.0, 0.1],
        [0.9, 0.0, 0.1, 1.0],
    ];
    println!(
        "symmetric contrastive loss: {:.4} -> {:.4}",
        initial.loss()?,
        model.loss()?
    );
    println!(
        "local perturbed-image recall@1: {:.0}%",
        100.0 * model.recall_at_one(&evaluation)?
    );
    let scores = model.scores()?;
    for (image, row) in scores.into_iter().enumerate() {
        let best = row
            .into_iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
            .ok_or("no captions")?;
        println!("image {image} retrieves {:?}", NAMES[best]);
    }
    println!(
        "The three-pair fixture verifies alignment mechanics, not open-vocabulary understanding."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_improves_loss_and_retrieval() -> Result<(), &'static str> {
        let initial = initial_model();
        let model = initial.train(300, 0.03)?;
        assert!(model.loss()? < initial.loss()? * 0.25);
        assert_ne!(model.image, initial.image);
        assert_ne!(model.text, initial.text);
        assert_eq!(model.recall_at_one(&IMAGES)?, 1.0);
        Ok(())
    }

    #[test]
    fn invalid_embeddings_are_rejected() {
        let zero = DualEncoder {
            image: [[0.0; 4]; 2],
            text: [[0.0; 3]; 2],
        };
        assert!(zero.image_embedding(IMAGES[0]).is_err());
        assert!(initial_model().image_embedding([f64::NAN; 4]).is_err());
        assert_eq!(caption_features("two vertical pixels"), Ok([1.0, 0.0, 0.0]));
        assert!(caption_features("unknown description").is_err());
    }
}
