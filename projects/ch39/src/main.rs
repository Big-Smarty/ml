use ch36::{Config, Decoder};
use ch38::{TrainConfig, Trainer};
use std::{collections::HashSet, error::Error, fs, path::PathBuf, time::Instant};

const TINY: &str = include_str!("../data/tiny.txt");
const VALIDATION: &str = include_str!("../data/validation.txt");

fn default_config() -> Config {
    Config {
        vocab_size: 256,
        context: 12,
        width: 16,
        heads: 2,
        layers: 1,
        ff_width: 32,
    }
}

#[derive(Default)]
struct Args {
    corpus: Option<PathBuf>,
    validation: Option<PathBuf>,
    steps: Option<u64>,
    checkpoint: Option<PathBuf>,
    resume: Option<PathBuf>,
    large_info: bool,
    large_smoke: bool,
    large: bool,
    gpu: bool,
    generate: Option<usize>,
}

fn args() -> Result<Args, Box<dyn Error>> {
    let mut out = Args::default();
    let mut values = std::env::args().skip(1);
    while let Some(value) = values.next() {
        match value.as_str() {
            "--corpus" => out.corpus = Some(values.next().ok_or("--corpus needs a path")?.into()),
            "--validation" => {
                out.validation = Some(values.next().ok_or("--validation needs a path")?.into())
            }
            "--steps" => out.steps = Some(values.next().ok_or("--steps needs a number")?.parse()?),
            "--checkpoint" => {
                out.checkpoint = Some(values.next().ok_or("--checkpoint needs a path")?.into())
            }
            "--resume" => out.resume = Some(values.next().ok_or("--resume needs a path")?.into()),
            "--large-info" => out.large_info = true,
            "--large-smoke" => out.large_smoke = true,
            "--large" => out.large = true,
            "--gpu" => out.gpu = true,
            "--generate" => {
                out.generate = Some(values.next().ok_or("--generate needs a number")?.parse()?)
            }
            _ => return Err(format!("unknown argument {value}").into()),
        }
    }
    Ok(out)
}

fn evaluate(
    model: &Decoder,
    validation_data: &[usize],
    targets_per_block: usize,
    target_budget: usize,
) -> Result<f32, Box<dyn Error>> {
    if targets_per_block == 0 || targets_per_block > model.config().context || target_budget == 0 {
        return Err("validation needs a positive model-compatible context and token budget".into());
    }
    if validation_data.len() < 2 {
        return Err("validation text needs at least two bytes".into());
    }
    let (mut weighted, mut count, mut start) = (0.0, 0usize, 0usize);
    let available = (validation_data.len() - 1).min(target_budget);
    while start < available {
        let targets_in_block = targets_per_block.min(available - start);
        let loss = model.loss(
            &validation_data[start..start + targets_in_block],
            &validation_data[start + 1..start + targets_in_block + 1],
        )?;
        weighted += loss * targets_in_block as f32;
        count += targets_in_block;
        start += targets_in_block;
    }
    Ok(weighted / count as f32)
}

fn has_overlap(train: &[u8], valid: &[u8]) -> bool {
    if train == valid {
        return true;
    }
    let (small, large) = if train.len() <= valid.len() {
        (train, valid)
    } else {
        (valid, train)
    };
    if small.len() < 32 {
        return false;
    }
    let windows: HashSet<&[u8]> = small.windows(32).collect();
    large.windows(32).any(|window| windows.contains(window))
}

fn main() -> Result<(), Box<dyn Error>> {
    let a = args()?;
    let large = Config::approximately_15m();
    if a.large_info || a.large_smoke {
        println!("large configuration: width=384 layers=8 heads=8 ff=1536 context=128 vocab=256");
        println!("exact trainable parameters: {}", large.parameter_count()?);
        println!("parameter bytes (f32): {}", large.parameter_count()? * 4);
        println!(
            "AdamW model+m+v bytes, excluding activations: {}",
            large.parameter_count()? * 12
        );
        if a.large_smoke {
            let started = Instant::now();
            let model = Decoder::new(large, 39)?;
            let logits = model.forward(&[84])?;
            println!(
                "large one-token CPU forward produced {} logits in {:?}",
                logits.len(),
                started.elapsed()
            );
        }
        return Ok(());
    }
    if a.corpus.is_some() != a.validation.is_some() {
        return Err(
            "custom training requires both --corpus and --validation document files".into(),
        );
    }
    if a.large && a.steps.is_none() {
        return Err("--large requires an explicit --steps budget".into());
    }
    let (raw_train_data, raw_validation_data) =
        if let (Some(train), Some(valid)) = (&a.corpus, &a.validation) {
            (fs::read(train)?, fs::read(valid)?)
        } else {
            (TINY.as_bytes().to_vec(), VALIDATION.as_bytes().to_vec())
        };
    if raw_train_data.len() < 2 || raw_validation_data.len() < 2 {
        return Err("training and validation documents need at least two bytes".into());
    }
    if has_overlap(&raw_train_data, &raw_validation_data) {
        return Err("training and validation documents share an exact 32-byte passage".into());
    }
    let train_data = raw_train_data
        .iter()
        .map(|&byte| byte as usize)
        .collect::<Vec<_>>();
    let validation_data = raw_validation_data
        .iter()
        .map(|&byte| byte as usize)
        .collect::<Vec<_>>();
    let steps = a.steps.unwrap_or(40);
    let mut trainer = if let Some(path) = &a.resume {
        let loaded = Trainer::load(path)?;
        if a.large && loaded.model.config() != large {
            return Err("--large checkpoint dimensions do not match".into());
        }
        loaded
    } else {
        let config = if a.large { large } else { default_config() };
        let mut train_config = TrainConfig::default();
        train_config.total_steps = steps.max(1);
        train_config.warmup_steps = train_config.warmup_steps.min(train_config.total_steps);
        Trainer::new(Decoder::new(config, 39)?, train_config, 390)?
    };
    let evaluation_target_budget = if a.large { 16 } else { usize::MAX };
    let targets_per_microbatch = if a.large {
        8
    } else {
        trainer.model.config().context
    };
    let starting_step = trainer.step;
    let before = evaluate(
        &trainer.model,
        &validation_data,
        trainer.model.config().context,
        evaluation_target_budget,
    )?;
    let mut processed_targets = 0usize;
    #[cfg(feature = "gpu")]
    let gpu = if a.gpu { Some(ch30::Gpu::new()?) } else { None };
    #[cfg(not(feature = "gpu"))]
    if a.gpu {
        return Err("rebuild with --features gpu before using --gpu".into());
    }
    let started = Instant::now();
    for _ in 0..steps {
        #[cfg(feature = "gpu")]
        if let Some(ref device) = gpu {
            trainer.train_accumulated_gpu(&train_data, targets_per_microbatch, 2, device)?;
        } else {
            trainer.train_accumulated(&train_data, targets_per_microbatch, 2)?;
        }
        #[cfg(not(feature = "gpu"))]
        trainer.train_accumulated(&train_data, targets_per_microbatch, 2)?;
        processed_targets = processed_targets
            .checked_add(targets_per_microbatch * 2)
            .ok_or("token counter overflow")?;
    }
    let elapsed = started.elapsed();
    let after = evaluate(
        &trainer.model,
        &validation_data,
        trainer.model.config().context,
        evaluation_target_budget,
    )?;
    let prompt = b"The ".iter().map(|&b| b as usize).collect::<Vec<_>>();
    let generated =
        trainer
            .model
            .generate(&prompt, a.generate.unwrap_or(80), &mut trainer.rng, 0.8)?;
    let bytes = generated.iter().map(|&x| x as u8).collect::<Vec<_>>();
    println!("trainable parameters: {}", trainer.model.parameter_count());
    println!("validation cross-entropy: {before:.3} -> {after:.3}");
    if a.large {
        println!(
            "large smoke evaluation used the first {evaluation_target_budget} held-out target bytes"
        );
    }
    let seconds = elapsed.as_secs_f64();
    println!(
        "optimizer steps: {starting_step} -> {}; processed {processed_targets} target tokens in {seconds:.3}s ({:.1} target tokens/s)",
        trainer.step,
        if seconds > 0.0 {
            processed_targets as f64 / seconds
        } else {
            0.0
        }
    );
    println!(
        "model+Adam moments: {} bytes; activations depend on sequence and layers",
        trainer.memory_bytes()
    );
    println!("sample: {}", String::from_utf8_lossy(&bytes));
    if let Some(path) = a.checkpoint.as_deref() {
        trainer.save(path)?;
        println!("saved resumable checkpoint to {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ch36::Rng;
    #[test]
    fn published_parameter_counts_are_exact() {
        let tiny = default_config();
        assert_eq!(tiny.vocab_size, 256);
        assert_eq!(tiny.parameter_count().unwrap(), 10_896);
        let large = Config::approximately_15m();
        assert_eq!(large.vocab_size, 256);
        assert_eq!(large.parameter_count().unwrap(), 14_442_496);
    }
    #[test]
    fn tiny_run_trains_saves_loads_and_generates() {
        let train_data = TINY
            .as_bytes()
            .iter()
            .map(|&b| b as usize)
            .collect::<Vec<_>>();
        let model = Decoder::new(
            Config {
                vocab_size: 256,
                context: 8,
                width: 8,
                heads: 2,
                layers: 1,
                ff_width: 12,
            },
            1,
        )
        .unwrap();
        let mut trainer = Trainer::new(model, TrainConfig::default(), 2).unwrap();
        let before = trainer
            .model
            .loss(&train_data[..8], &train_data[1..9])
            .unwrap();
        for _ in 0..12 {
            trainer.train_accumulated(&train_data, 8, 1).unwrap();
        }
        assert!(
            trainer
                .model
                .loss(&train_data[..8], &train_data[1..9])
                .unwrap()
                < before
        );
        let p = std::env::temp_dir().join(format!("ch39-{}.bin", std::process::id()));
        trainer.save(&p).unwrap();
        let mut loaded = Trainer::load(&p).unwrap();
        let mut changed = Trainer::load(&p).unwrap();
        let mut changed_data = train_data.clone();
        changed_data[0] ^= 1;
        assert!(changed.train_accumulated(&changed_data, 8, 1).is_err());
        let continued_loss = trainer.train_accumulated(&train_data, 8, 1).unwrap();
        let resumed_loss = loaded.train_accumulated(&train_data, 8, 1).unwrap();
        assert_eq!(continued_loss.to_bits(), resumed_loss.to_bits());
        assert_eq!(trainer.model.parameters(), loaded.model.parameters());
        let mut rng = Rng::new(3);
        assert_eq!(
            loaded
                .model
                .generate(&[b'T' as usize], 4, &mut rng, 1.0)
                .unwrap()
                .len(),
            5
        );
        fs::remove_file(p).unwrap();
    }
    #[test]
    fn overlap_and_tail_evaluation_are_checked() {
        assert!(has_overlap(&[b'x'; 40], &[b'x'; 40]));
        let model = Decoder::new(
            Config {
                vocab_size: 256,
                context: 4,
                width: 4,
                heads: 1,
                layers: 1,
                ff_width: 6,
            },
            2,
        )
        .unwrap();
        let data = [1, 2, 3, 4, 5, 6, 7];
        let expected = (4.0 * model.loss(&data[..4], &data[1..5]).unwrap()
            + 2.0 * model.loss(&data[4..6], &data[5..7]).unwrap())
            / 6.0;
        assert_eq!(evaluate(&model, &data, 4, usize::MAX).unwrap(), expected);
        assert!(evaluate(&model, &data, 0, 4).is_err());
        assert!(evaluate(&model, &data, 4, 0).is_err());
        assert!(has_overlap(
            b"prefix abcdefghijklmnopqrstuvwxyz012345 suffix",
            b"abcdefghijklmnopqrstuvwxyz012345 different"
        ));
        assert!(evaluate(&model, &[1, 2, 3], 4, usize::MAX)
            .unwrap()
            .is_finite());
    }
}
