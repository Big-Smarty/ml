use ch36::{Config, Decoder};
use ch38::{TrainConfig, Trainer};
use std::{collections::HashSet, error::Error, fs, path::PathBuf, time::Instant};

const TINY: &str = include_str!("../data/tiny.txt");
const VALIDATION: &str = include_str!("../data/validation.txt");

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
    data: &[usize],
    context: usize,
    token_budget: usize,
) -> Result<f32, Box<dyn Error>> {
    if context == 0 || context > model.config().context || token_budget == 0 {
        return Err("validation needs a positive model-compatible context and token budget".into());
    }
    if data.len() < 2 {
        return Err("validation text needs at least two bytes".into());
    }
    let (mut weighted, mut count, mut start) = (0.0, 0usize, 0usize);
    let available = (data.len() - 1).min(token_budget);
    while start < available {
        let tokens = context.min(available - start);
        let loss = model.loss(
            &data[start..start + tokens],
            &data[start + 1..start + tokens + 1],
        )?;
        weighted += loss * tokens as f32;
        count += tokens;
        start += tokens;
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
    let (raw, raw_valid) = if let (Some(train), Some(valid)) = (&a.corpus, &a.validation) {
        (fs::read(train)?, fs::read(valid)?)
    } else {
        (TINY.as_bytes().to_vec(), VALIDATION.as_bytes().to_vec())
    };
    if raw.len() < 2 || raw_valid.len() < 2 {
        return Err("training and validation documents need at least two bytes".into());
    }
    if has_overlap(&raw, &raw_valid) {
        return Err("training and validation documents share an exact 32-byte passage".into());
    }
    let train = raw.iter().map(|&b| b as usize).collect::<Vec<_>>();
    let valid = raw_valid.iter().map(|&b| b as usize).collect::<Vec<_>>();
    let steps = a.steps.unwrap_or(40);
    let mut trainer = if let Some(path) = &a.resume {
        let loaded = Trainer::load(path)?;
        if a.large && loaded.model.config() != large {
            return Err("--large checkpoint dimensions do not match".into());
        }
        loaded
    } else {
        let config = if a.large {
            large
        } else {
            Config {
                vocab_size: 256,
                context: 12,
                width: 16,
                heads: 2,
                layers: 1,
                ff_width: 32,
            }
        };
        let mut train_config = TrainConfig::default();
        train_config.total_steps = steps.max(1);
        train_config.warmup_steps = train_config.warmup_steps.min(train_config.total_steps);
        Trainer::new(Decoder::new(config, 39)?, train_config, 390)?
    };
    let eval_budget = if a.large { 16 } else { usize::MAX };
    let train_tokens = if a.large {
        8
    } else {
        trainer.model.config().context
    };
    let before = evaluate(
        &trainer.model,
        &valid,
        trainer.model.config().context,
        eval_budget,
    )?;
    let mut seen = 0usize;
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
            trainer.train_accumulated_gpu(&train, train_tokens, 2, device)?;
        } else {
            trainer.train_accumulated(&train, train_tokens, 2)?;
        }
        #[cfg(not(feature = "gpu"))]
        trainer.train_accumulated(&train, train_tokens, 2)?;
        seen = seen
            .checked_add(train_tokens * 2)
            .ok_or("token counter overflow")?;
    }
    let elapsed = started.elapsed();
    let after = evaluate(
        &trainer.model,
        &valid,
        trainer.model.config().context,
        eval_budget,
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
        println!("large smoke evaluation used the first {eval_budget} held-out target bytes");
    }
    let seconds = elapsed.as_secs_f64();
    println!(
        "processed {seen} tokens in {seconds:.3}s ({:.1} tokens/s)",
        if seconds > 0.0 {
            seen as f64 / seconds
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
    fn tiny_run_trains_saves_loads_and_generates() {
        let data = TINY
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
            .loss_and_grad(&data[..8], &data[1..9])
            .unwrap()
            .loss;
        for _ in 0..12 {
            trainer.train_accumulated(&data, 8, 1).unwrap();
        }
        assert!(
            trainer
                .model
                .loss_and_grad(&data[..8], &data[1..9])
                .unwrap()
                .loss
                < before
        );
        let p = std::env::temp_dir().join(format!("ch39-{}.bin", std::process::id()));
        trainer.save(&p).unwrap();
        let loaded = Trainer::load(&p).unwrap();
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
