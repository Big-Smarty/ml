// A trainable dual encoder for three course-authored image-caption pairs.

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
    image_weights: [[f64; 4]; 2],
    text_weights: [[f64; 3]; 2],
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
    fn image_embedding(&self, pixels: [f64; 4]) -> Result<[f64; 2], &'static str> {
        if pixels.iter().any(|v| !v.is_finite()) {
            return Err("image pixels must be finite");
        }
        normalize(
            self.image_weights
                .map(|row| row.iter().zip(pixels).map(|(w, x)| w * x).sum()),
        )
    }

    fn text_embedding(&self, words: [f64; 3]) -> Result<[f64; 2], &'static str> {
        if words.iter().any(|v| !v.is_finite()) {
            return Err("text features must be finite");
        }
        normalize(
            self.text_weights
                .map(|row| row.iter().zip(words).map(|(w, x)| w * x).sum()),
        )
    }

    fn caption_embedding(&self, caption: &str) -> Result<[f64; 2], &'static str> {
        self.text_embedding(caption_features(caption)?)
    }

    fn logits(
        &self,
        images: &[[f64; 4]; 3],
        captions: &[&str; 3],
        temperature: f64,
    ) -> Result<[[f64; 3]; 3], &'static str> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err("temperature must be finite and positive");
        }
        let logits = self
            .similarities(images, captions)?
            .map(|row| row.map(|similarity| similarity / temperature));
        if logits.iter().flatten().any(|logit| !logit.is_finite()) {
            return Err("temperature produces nonfinite logits");
        }
        Ok(logits)
    }

    fn parameters(&self) -> [f64; 14] {
        let mut parameters = [0.0; 14];
        for (slot, value) in parameters.iter_mut().zip(
            self.image_weights
                .into_iter()
                .flatten()
                .chain(self.text_weights.into_iter().flatten()),
        ) {
            *slot = value;
        }
        parameters
    }

    fn from_parameters(parameters: [f64; 14]) -> Self {
        Self {
            image_weights: [
                [parameters[0], parameters[1], parameters[2], parameters[3]],
                [parameters[4], parameters[5], parameters[6], parameters[7]],
            ],
            text_weights: [
                [parameters[8], parameters[9], parameters[10]],
                [parameters[11], parameters[12], parameters[13]],
            ],
        }
    }

    fn train(
        self,
        images: &[[f64; 4]; 3],
        captions: &[&str; 3],
        steps: usize,
        learning_rate: f64,
        temperature: f64,
    ) -> Result<Self, &'static str> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        let mut parameters = self.parameters();
        for _ in 0..steps {
            let mut gradient = [0.0; 14];
            for i in 0..parameters.len() {
                let mut plus = parameters;
                let mut minus = parameters;
                plus[i] += 1e-5;
                minus[i] -= 1e-5;
                gradient[i] = (Self::from_parameters(plus).loss(images, captions, temperature)?
                    - Self::from_parameters(minus).loss(images, captions, temperature)?)
                    / 2e-5;
            }
            for (parameter, slope) in parameters.iter_mut().zip(gradient) {
                *parameter -= learning_rate * slope;
            }
        }
        let model = Self::from_parameters(parameters);
        model.loss(images, captions, temperature)?;
        Ok(model)
    }
}

fn cross_entropy_from_logits(logits: [f64; 3], target: usize) -> f64 {
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (maximum - logits[target])
        + logits
            .iter()
            .map(|logit| (logit - maximum).exp())
            .sum::<f64>()
            .ln()
}

fn initial_model() -> DualEncoder {
    DualEncoder {
        image_weights: [[0.30, -0.20, 0.10, 0.25], [-0.15, 0.35, 0.20, -0.10]],
        text_weights: [[0.20, -0.30, 0.10], [0.25, 0.15, -0.20]],
    }
}

fn demo() -> Result<(), Box<dyn std::error::Error>> {
    let initial = initial_model();
    let model = initial.train(&IMAGES, &NAMES, 300, 0.03, TEMPERATURE)?;
    let evaluation = [
        [0.9, 0.1, 1.0, 0.0],
        [1.0, 0.9, 0.0, 0.1],
        [0.9, 0.0, 0.1, 1.0],
    ];
    println!(
        "selected alignment objective: {:.4} -> {:.4}",
        initial.loss(&IMAGES, &NAMES, TEMPERATURE)?,
        model.loss(&IMAGES, &NAMES, TEMPERATURE)?
    );
    println!(
        "local perturbed-image recall@1: {:.0}%",
        100.0 * model.recall_at_one(&evaluation, &NAMES)?
    );
    let similarities = model.similarities(&IMAGES, &NAMES)?;
    for (image, row) in similarities.into_iter().enumerate() {
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
