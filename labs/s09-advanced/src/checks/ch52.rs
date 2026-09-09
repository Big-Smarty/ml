pub fn check() -> Result<(), String> {
    crate::ensure(
        normalize([3.0, 4.0])? == [0.6, 0.8],
        "normalization must remove length",
    )?;
    let initial = initial_model();
    let logits = initial.logits(&IMAGES, &NAMES, TEMPERATURE)?;
    let expected = (0..3)
        .map(|i| {
            cross_entropy_from_logits(logits[i], i)
                + cross_entropy_from_logits([logits[0][i], logits[1][i], logits[2][i]], i)
        })
        .sum::<f64>()
        / 6.0;
    let actual = initial.loss(&IMAGES, &NAMES, TEMPERATURE)?;
    crate::ensure(crate::close(actual,expected),&format!("symmetric objective: actual {actual:.8}, expected {expected:.8}; inspect the column axis and 2N denominator"))?;
    let model = initial.train(&IMAGES, &NAMES, 300, 0.03, TEMPERATURE)?;
    let fresh = [
        [0.9, 0.1, 1.0, 0.0],
        [1.0, 0.9, 0.0, 0.1],
        [0.9, 0.0, 0.1, 1.0],
    ];
    crate::ensure(
        model.recall_at_one(&fresh, &NAMES)? == 1.0
            && model.image_weights != initial.image_weights
            && model.text_weights != initial.text_weights,
        "both encoders must train and retrieve perturbed local inputs",
    )?;
    crate::ensure(
        model.recall_at_one(&IMAGES, &[NAMES[0]; 3])? == 1.0 / 3.0,
        "duplicate candidates expose the declared last-index tie policy",
    )?;
    crate::ensure(
        normalize([0.0; 2]).is_err() && caption_features("unknown").is_err(),
        "zero and unknown-feature boundaries must reject",
    )
}

pub fn run(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("this experiment takes no extra arguments".into());
    }
    demo().map_err(|e| e.to_string())
}
