// Supplied checkpoint I/O, state validation, CPU reference optimizer and its regression tests.
// The learner algorithms enter through advance() below; the original remains an oracle.
include!("../../../projects/ch38/src/lib.rs");

#[derive(Debug)]
pub struct Proposal {
    pub parameters: Vec<f32>,
    pub first: Vec<f32>,
    pub second: Vec<f32>,
}
pub type Update =
    fn(&[f32], &[f32], &[f32], &[f32], TrainConfig, u64) -> crate::LabResult<Proposal>;
pub type Combine = fn(&[Gradients]) -> crate::LabResult<Gradients>;
pub type Rate = fn(TrainConfig, u64) -> f32;
pub fn validate_update(
    p: &[f32],
    m: &[f32],
    v: &[f32],
    g: &[f32],
    c: TrainConfig,
    step: u64,
) -> crate::LabResult {
    validate_train(c).map_err(str::to_owned)?;
    crate::ensure(
        !p.is_empty()
            && m.len() == p.len()
            && v.len() == p.len()
            && g.len() == p.len()
            && p.iter().chain(m).chain(v).chain(g).all(|x| x.is_finite())
            && v.iter().all(|x| *x >= 0.)
            && step < u64::MAX,
        "invalid optimizer input",
    )
}
pub fn validate_batches(batches: &[Gradients]) -> crate::LabResult {
    crate::ensure(!batches.is_empty(), "need at least one microbatch")?;
    let n = batches[0].values.len();
    crate::ensure(
        n > 0
            && batches.iter().all(|b| {
                b.tokens > 0
                    && b.values.len() == n
                    && b.loss.is_finite()
                    && b.values.iter().all(|v| v.is_finite())
            }),
        "invalid gradient batch",
    )
}
pub fn windows(
    documents: &[Vec<usize>],
    context: usize,
) -> crate::LabResult<Vec<(&[usize], &[usize])>> {
    crate::ensure(
        context > 0 && context <= 128 && !documents.is_empty(),
        "need documents and context in 1..=128",
    )?;
    let mut out = Vec::new();
    for doc in documents {
        crate::ensure(
            doc.len() >= 2 && doc.iter().all(|&b| b < 256),
            "every byte document needs at least two IDs",
        )?;
        let mut start = 0;
        while start < doc.len() - 1 {
            let n = context.min(doc.len() - 1 - start);
            out.push((&doc[start..start + n], &doc[start + 1..start + n + 1]));
            start += n;
        }
    }
    Ok(out)
}
pub fn advance(
    trainer: &mut Trainer,
    documents: &[Vec<usize>],
    context: usize,
    micro: usize,
    update: Update,
    combine: Combine,
) -> crate::LabResult<f32> {
    trainer.validate_state().map_err(|e| e.to_string())?;
    crate::ensure(
        micro > 0 && micro <= 16 && context <= trainer.model.config().context,
        "invalid microbatch count/context",
    )?;
    let examples = windows(documents, context)?;
    // Include document lengths so identical concatenation with different boundaries has a different identity.
    let identity: Vec<usize> = documents
        .iter()
        .flat_map(|d| std::iter::once(d.len() + 256).chain(d.iter().copied()))
        .collect();
    let id = fingerprint(&identity);
    crate::ensure(
        trainer.data_fingerprint == 0 || trainer.data_fingerprint == id,
        "training documents/boundaries differ from checkpoint",
    )?;
    let mut cursor = trainer.data_cursor % examples.len();
    let mut batches = Vec::new();
    let mut rng = trainer.rng;
    for _ in 0..micro {
        let (x, y) = examples[cursor];
        batches.push(
            trainer
                .model
                .loss_and_gradient(x, y)
                .map_err(|e| e.to_string())?,
        );
        cursor = (cursor + 1) % examples.len();
        rng.next_u64();
    }
    let gradient = combine(&batches)?;
    let proposal = update(
        trainer.model.parameters(),
        &trainer.m,
        &trainer.v,
        &gradient.values,
        trainer.config,
        trainer.step,
    )?;
    crate::ensure(
        [
            proposal.parameters.len(),
            proposal.first.len(),
            proposal.second.len(),
        ]
        .iter()
        .all(|&n| n == trainer.model.parameter_count())
            && proposal
                .parameters
                .iter()
                .chain(&proposal.first)
                .chain(&proposal.second)
                .all(|v| v.is_finite())
            && proposal.second.iter().all(|v| *v >= 0.),
        "proposed update is invalid; state unchanged",
    )?;
    let step = trainer
        .step
        .checked_add(1)
        .ok_or("optimizer step overflow")?;
    trainer
        .model
        .parameters_mut()
        .copy_from_slice(&proposal.parameters);
    trainer.m = proposal.first;
    trainer.v = proposal.second;
    trainer.step = step;
    trainer.data_cursor = cursor;
    trainer.rng = rng;
    trainer.data_fingerprint = id;
    Ok(gradient.loss)
}
pub fn tiny_trainer() -> crate::LabResult<Trainer> {
    let config = Config {
        vocab_size: 256,
        context: 4,
        width: 8,
        heads: 2,
        layers: 1,
        ff_width: 12,
    };
    Trainer::new(
        Decoder::new(config, 38).map_err(|e| e.to_string())?,
        TrainConfig {
            peak_learning_rate: 0.02,
            min_learning_rate: 0.002,
            warmup_steps: 2,
            total_steps: 40,
            ..TrainConfig::default()
        },
        99,
    )
    .map_err(str::to_owned)
}
pub fn run_with(args: &[String], update: Update, combine: Combine) -> crate::LabResult {
    crate::no_args(args)?;
    let docs = vec![
        b"abcabcabc".iter().map(|&b| b as usize).collect(),
        b"bcabca".iter().map(|&b| b as usize).collect(),
    ];
    let mut trainer = tiny_trainer()?;
    let first = advance(&mut trainer, &docs, 4, 2, update, combine)?;
    let mut last = first;
    for _ in 0..23 {
        last = advance(&mut trainer, &docs, 4, 2, update, combine)?;
    }
    println!(
        "actual decoder training: data loss={first:.5}->{last:.5}; step={} persistent bytes={}",
        trainer.step,
        trainer.memory_bytes()
    );
    resume_check(&trainer, &docs, update, combine)?;
    println!("checkpoint next update matches, including moments/cursor/RNG/step");
    Ok(())
}
pub fn resume_check(
    trainer: &Trainer,
    docs: &[Vec<usize>],
    update: Update,
    combine: Combine,
) -> crate::LabResult {
    let path = std::env::temp_dir().join(format!(
        "s07-resume-{}-{}.bin",
        std::process::id(),
        trainer.step
    ));
    trainer.save(&path).map_err(|e| e.to_string())?;
    let mut restored = Trainer::load(&path).map_err(|e| e.to_string())?;
    std::fs::remove_file(path).map_err(|e| e.to_string())?;
    let mut continued = trainer.clone();
    let a = advance(&mut continued, docs, 4, 2, update, combine)?;
    let b = advance(&mut restored, docs, 4, 2, update, combine)?;
    crate::ensure(
        a.to_bits() == b.to_bits()
            && continued.model.parameters() == restored.model.parameters()
            && continued.m == restored.m
            && continued.v == restored.v
            && continued.rng == restored.rng
            && continued.step == restored.step
            && continued.data_cursor == restored.data_cursor
            && continued.learning_rate() == restored.learning_rate(),
        "resume next-step state mismatch",
    )
}
pub fn check_with(update: Update, combine: Combine, rate: Rate) -> crate::LabResult {
    let cfg = TrainConfig {
        peak_learning_rate: 1.,
        min_learning_rate: 0.1,
        warmup_steps: 4,
        total_steps: 12,
        ..TrainConfig::default()
    };
    for (step, expected) in [(0, 0.25), (3, 1.), (4, 1.), (8, 0.55), (12, 0.1), (20, 0.1)] {
        crate::ensure(
            (rate(cfg, step) - expected).abs() < 1e-6,
            format!(
                "goal: schedule at completed step {step}: expected {expected}, got {}",
                rate(cfg, step)
            ),
        )?;
    }
    let batches = [
        Gradients {
            loss: 2.,
            tokens: 4,
            values: vec![3., 0.],
        },
        Gradients {
            loss: 5.,
            tokens: 2,
            values: vec![0., 6.],
        },
    ];
    let combined = combine(&batches)?;
    crate::ensure(
        combined.tokens == 6 && combined.values == [2., 2.] && combined.loss == 3.,
        "goal: combine unequal microbatch means weighted by target count",
    )?;
    let cfg = TrainConfig {
        peak_learning_rate: 0.001,
        min_learning_rate: 0.001,
        warmup_steps: 0,
        total_steps: 20,
        clip_norm: 1.,
        ..TrainConfig::default()
    };
    let proposal = update(&[2., -1.], &[0., 0.], &[0., 0.], &[3., 4.], cfg, 0)?;
    crate::ensure(
        (proposal.parameters[0] - 1.99898).abs() < 2e-6
            && (proposal.first[0] - 0.06).abs() < 1e-6
            && (proposal.second[1] - 0.00064).abs() < 1e-7,
        "goal: global clip [3,4] to [0.6,0.8], then bias-corrected AdamW with separate decay",
    )?;
    let mut reference_stream = tiny_trainer()?;
    reference_stream
        .train_accumulated(&[97, 98, 99, 97, 98, 99, 97, 98, 99], 4, 2)
        .map_err(|e| e.to_string())?;
    let mut trainer = tiny_trainer()?;
    trainer.config = cfg;
    let n = trainer.model.parameter_count();
    let g: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.03 - 0.08).collect();
    for _ in 0..3 {
        let proposal = update(
            trainer.model.parameters(),
            &trainer.m,
            &trainer.v,
            &g,
            cfg,
            trainer.step,
        )?;
        trainer.adamw(&mut g.clone()).map_err(|e| e.to_string())?;
        crate::ensure(
            proposal
                .parameters
                .iter()
                .zip(trainer.model.parameters())
                .all(|(a, b)| (a - b).abs() < 2e-6)
                && proposal
                    .first
                    .iter()
                    .zip(&trainer.m)
                    .all(|(a, b)| (a - b).abs() < 1e-6),
            "goal: successive moments/parameters disagree with reference AdamW",
        )?;
    }
    crate::ensure(
        update(&[1.], &[0.], &[0.], &[f32::NAN], cfg, 0).is_err(),
        "goal: invalid gradient must return error",
    )?;
    let docs = vec![vec![97, 98, 99, 97, 98, 99, 97], vec![98, 99, 97]];
    let mut trainer = tiny_trainer()?;
    advance(&mut trainer, &docs, 4, 2, update, combine)?;
    resume_check(&trainer, &docs, update, combine)?;
    let mut changed = docs.clone();
    changed[0][0] = 100;
    crate::ensure(
        advance(&mut trainer, &changed, 4, 2, update, combine).is_err(),
        "goal: changed corpus accepted after checkpoint binding",
    )?;
    println!(
        "38 goal passed: schedule, token-weighted accumulation, clipping, AdamW and exact resume"
    );
    Ok(())
}
#[cfg(test)]
mod lab_tests {
    #[test]
    fn working_sgd_and_complete_optimizer() {
        super::run_with(&[], crate::ch38::update, crate::ch38::combine).unwrap();
        super::check_with(
            crate::solutions::ch38::update,
            crate::solutions::ch38::combine,
            crate::solutions::ch38::rate,
        )
        .unwrap();
    }
    #[test]
    fn windows_never_cross_documents() {
        let docs = vec![vec![1, 2, 3, 4, 5, 6], vec![9, 8, 7]];
        let w = super::windows(&docs, 4).unwrap();
        assert_eq!(w.len(), 3);
        assert_eq!(w[1], (&[5][..], &[6][..]));
        assert_eq!(w[2], (&[9, 8][..], &[8, 7][..]));
    }
}
