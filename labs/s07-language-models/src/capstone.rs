//! Supplied bounded CLI, corpus checks, reporting, checkpoint I/O and training loop.
use crate::{
    ensure,
    training::{self, Trainer},
    LabResult,
};
use ::ch36::{Config, Decoder, Rng};
use std::{collections::HashSet, path::PathBuf, time::Instant};
pub type Train = fn(&mut Trainer, &[Vec<usize>]) -> LabResult<(f32, usize)>;
pub type Evaluate = fn(&Decoder, &[Vec<usize>]) -> LabResult<f32>;
pub fn targets(trainer: &Trainer, docs: &[Vec<usize>], micro: usize) -> LabResult<usize> {
    let windows = training::windows(docs, trainer.model.config().context)?;
    Ok((0..micro)
        .map(|i| windows[(trainer.data_cursor + i) % windows.len()].0.len())
        .sum())
}
pub fn evaluation_blocks(model: &Decoder, docs: &[Vec<usize>]) -> LabResult<Vec<(f32, usize)>> {
    training::windows(docs, model.config().context)?
        .into_iter()
        .map(|(x, y)| Ok((model.loss(x, y).map_err(|e| e.to_string())?, y.len())))
        .collect()
}
fn ids(bytes: &[u8]) -> Vec<usize> {
    bytes.iter().map(|&b| b as usize).collect()
}
fn corpus_pair(train: &[u8], valid: &[u8]) -> LabResult {
    ensure(
        train.len() >= 2 && valid.len() >= 2,
        "training and validation need at least two bytes",
    )?;
    ensure(train != valid, "training and validation are identical")?;
    let windows: HashSet<_> = train.windows(32).collect();
    ensure(
        !valid.windows(32).any(|w| windows.contains(w)),
        "exact 32-byte train/validation overlap; audit corpus",
    )
}
#[derive(Default)]
struct Args {
    steps: Option<usize>,
    context: Option<usize>,
    checkpoint: Option<PathBuf>,
    resume: Option<PathBuf>,
    corpus: Option<PathBuf>,
    validation: Option<PathBuf>,
    extended: bool,
    large_info: bool,
}
fn parse(args: &[String]) -> LabResult<Args> {
    let mut out = Args::default();
    let mut values = args.iter();
    while let Some(arg) = values.next() {
        match arg.as_str() {
        "--steps"=>out.steps=Some(values.next().ok_or("--steps needs N")?.parse::<usize>().map_err(|e|e.to_string())?),
        "--context"=>out.context=Some(values.next().ok_or("--context needs N")?.parse::<usize>().map_err(|e|e.to_string())?),
        "--checkpoint"=>out.checkpoint=Some(values.next().ok_or("--checkpoint needs PATH")?.into()),
        "--resume"=>out.resume=Some(values.next().ok_or("--resume needs PATH")?.into()),
        "--corpus"=>out.corpus=Some(values.next().ok_or("--corpus needs PATH")?.into()),
        "--validation"=>out.validation=Some(values.next().ok_or("--validation needs PATH")?.into()),
        "--extended"=>out.extended=true,"--large-info"=>out.large_info=true,
        "--gpu"=>return Err("unsupported in the bounded lab; explicit GPU training remains in projects/ch39 with --features gpu -- --gpu".into()),
        _=>return Err(format!("unknown option {arg}"))
    }
    }
    ensure(
        out.corpus.is_some() == out.validation.is_some(),
        "custom corpus requires --corpus and --validation",
    )?;
    ensure(
        out.steps.unwrap_or(40) <= if out.extended { 2000 } else { 200 },
        "more than 200 updates needs --extended; hard teaching ceiling 2000",
    )?;
    ensure(
        matches!(out.context, None | Some(4) | Some(8)),
        "choose --context 4 or 8",
    )?;
    Ok(out)
}
fn read(path: &std::path::Path) -> LabResult<Vec<u8>> {
    ensure(
        std::fs::metadata(path).map_err(|e| e.to_string())?.len() <= 65536,
        "local text limit 64 KiB; use an audited bounded sample",
    )?;
    std::fs::read(path).map_err(|e| e.to_string())
}
pub fn run_with(args: &[String], train: Train, evaluate: Evaluate) -> LabResult {
    let args = parse(args)?;
    if args.large_info {
        let c = Config::approximately_15m();
        let n = c.parameter_count().map_err(|e| e.to_string())?;
        println!("large dimensions={c:?}; parameters={n}; f32 weights={} bytes; weights+m+v={} bytes; no allocation",4*n,12*n);
        return Ok(());
    }
    let (raw, valid) = if let (Some(a), Some(b)) = (&args.corpus, &args.validation) {
        (read(a)?, read(b)?)
    } else {
        (
            include_bytes!("../data/train.txt").to_vec(),
            include_bytes!("../data/validation.txt").to_vec(),
        )
    };
    corpus_pair(&raw, &valid)?;
    let docs = vec![ids(&raw)];
    let validation = vec![ids(&valid)];
    let mut trainer = if let Some(path) = args.resume.as_deref() {
        ensure(
            std::fs::metadata(path).map_err(|e| e.to_string())?.len() <= 250_000,
            "checkpoint exceeds tiny lab file budget",
        )?;
        let loaded = Trainer::load(path).map_err(|e| e.to_string())?;
        ensure(
            loaded.model.config().vocab_size == 256
                && matches!(loaded.model.config().context, 4 | 8)
                && loaded.model.parameter_count() <= 20000,
            "checkpoint exceeds tiny byte lab architecture budget",
        )?;
        ensure(
            args.context.is_none() || args.context == Some(loaded.model.config().context),
            "resume context differs from checkpoint",
        )?;
        loaded
    } else {
        let mut t = training::tiny_trainer()?;
        if args.context == Some(8) {
            t.model = Decoder::new(
                Config {
                    context: 8,
                    ..t.model.config()
                },
                38,
            )
            .map_err(|e| e.to_string())?;
            t = Trainer::new(t.model, t.config, 99).map_err(str::to_owned)?;
        }
        t
    };
    let before = evaluate(&trainer.model, &validation)?;
    let train_before = evaluate(&trainer.model, &docs)?;
    let start_step = trainer.step;
    let start = Instant::now();
    let mut count = 0;
    for _ in 0..args.steps.unwrap_or(40) {
        let (_, targets) = train(&mut trainer, &docs)?;
        count += targets;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let after = evaluate(&trainer.model, &validation)?;
    let train_after = evaluate(&trainer.model, &docs)?;
    println!(
        "byte IDs 0..255; config={:?}; initial seeds for new runs=(38,99)",
        trainer.model.config()
    );
    println!("steps={start_step}->{} targets={count}; train nats/target={train_before:.6}->{train_after:.6}; validation={before:.6}->{after:.6}",trainer.step);
    println!(
        "validation bits/target={:.6}; block history restarts; train bytes={} validation bytes={}",
        after / std::f32::consts::LN_2,
        raw.len(),
        valid.len()
    );
    println!("parameters={} persistent bytes={}; training seconds={elapsed:.6}; targets/s={:.1}; CPU f32 (single run, includes gradient allocation, excludes eval/checkpoint)",trainer.model.parameter_count(),trainer.memory_bytes(),count as f64/elapsed.max(1e-9));
    let mut rng = Rng::new(390);
    let sample = trainer
        .model
        .generate(&ids(b"The "), 32, &mut rng, 0.8)
        .map_err(|e| e.to_string())?;
    let bytes: Vec<u8> = sample.iter().map(|&i| i as u8).collect();
    println!(
        "sample seed=390 temperature=0.8 raw={bytes:?}\ndisplay={:?}",
        String::from_utf8_lossy(&bytes)
    );
    // Validate the exact selected capstone train callback, not a different reference continuation.
    branch_resume(&trainer, &docs, train)?;
    println!("selected training algorithm resumes its exact next update");
    if let Some(path) = args.checkpoint {
        trainer.save(&path).map_err(|e| e.to_string())?;
        println!(
            "saved {} format CH38LM02; keep corpus and command with it",
            path.display()
        );
    }
    Ok(())
}
fn branch_resume(trainer: &Trainer, docs: &[Vec<usize>], train: Train) -> LabResult {
    let path = std::env::temp_dir().join(format!(
        "s07-capstone-{}-{}.bin",
        std::process::id(),
        trainer.step
    ));
    trainer.save(&path).map_err(|e| e.to_string())?;
    let mut b = Trainer::load(&path).map_err(|e| e.to_string())?;
    std::fs::remove_file(path).map_err(|e| e.to_string())?;
    let mut a = trainer.clone();
    let la = train(&mut a, docs)?;
    let lb = train(&mut b, docs)?;
    ensure(
        la == lb
            && a.model.parameters() == b.model.parameters()
            && a.step == b.step
            && a.data_cursor == b.data_cursor
            && a.rng == b.rng,
        "selected capstone train callback failed resume equivalence",
    )
}
pub fn check_with(train: Train, evaluate: Evaluate) -> LabResult {
    let mut model = training::tiny_trainer()?;
    let docs = vec![ids(b"abcdefg"), ids(b"xyzxy")];
    let expected = crate::solutions::ch39::evaluate(&model.model, &docs)?;
    let actual = evaluate(&model.model, &docs)?;
    ensure((actual-expected).abs()<1e-7,format!("goal: evaluation must weight every target including document tails: learner={actual:.8} expected={expected:.8}"))?;
    let mut reference = model.clone();
    let expected = crate::solutions::ch39::train(&mut reference, &docs)?;
    let actual = train(&mut model, &docs)?;
    ensure(actual==expected && model.model.parameters()==reference.model.parameters() && model.data_cursor==reference.data_cursor,"goal: compose document-safe windows, token-weighted microbatches and AdamW in one actual update")?;
    let docs = vec![ids(b"abcabcabcabc"), ids(b"bcabcabca")];
    let mut trainer = training::tiny_trainer()?;
    let original = trainer.model.parameters().to_vec();
    let before = evaluate(&trainer.model, &docs)?;
    for _ in 0..40 {
        train(&mut trainer, &docs)?;
    }
    let after = evaluate(&trainer.model, &docs)?;
    ensure(
        after < before - 1.,
        "goal: actual repeated-pattern decoder loss should fall by more than 1 nat",
    )?;
    for name in [
        "token_embedding",
        "position_embedding",
        "layer.0.qkv_weight",
        "layer.0.ff1_weight",
        "layer.0.ln1_gain",
        "output_weight",
    ] {
        let span = trainer
            .model
            .parameter_spans()
            .into_iter()
            .find(|s| s.name == name)
            .ok_or("missing trainable family")?;
        ensure(
            trainer.model.parameters()[span.start..span.end] != original[span.start..span.end],
            format!("goal: {name} remained frozen"),
        )?;
    }
    branch_resume(&trainer, &docs, train)?;
    let mut changed = docs.clone();
    changed[1][0] = 100;
    ensure(
        train(&mut trainer, &changed).is_err(),
        "goal: checkpoint accepted changed document identity",
    )?;
    let mut rng = Rng::new(39);
    ensure(
        trainer
            .model
            .generate(&[97], 8, &mut rng, 0.8)
            .map_err(|e| e.to_string())?
            .len()
            == 9,
        "goal: autoregressive byte generation",
    )?;
    println!("39 goal passed: train/evaluate full decoder, all families, tails, data binding, generation and resume; pattern loss={before:.5}->{after:.5}");
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn complete_capstone_and_cli_boundaries() {
        super::check_with(
            crate::solutions::ch39::train,
            crate::solutions::ch39::evaluate,
        )
        .unwrap();
        assert!(super::parse(&["--steps".into(), "201".into()]).is_err());
        assert!(super::corpus_pair(b"same", b"same").is_err());
        assert!(super::parse(&["--corpus".into(), "missing".into()]).is_err());
    }
}
