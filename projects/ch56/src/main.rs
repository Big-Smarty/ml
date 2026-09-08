use ch56::{serve_once, Config, Model, Rng, Trainer};
use std::{io::Read, path::Path};

// Original course-authored ASCII fixtures. The held-out sentence is never trained on.
const TRAIN: &[u8] = b"rust learns patterns. experts route tokens. rust predicts words. experts share work. rust learns words. experts route patterns. ";
const HELD_OUT: &[u8] = b"rust predicts patterns. experts learn words. ";

fn initialized(experts: usize) -> Result<Trainer, Box<dyn std::error::Error>> {
    let dense = Model::new(Config::tiny(1), 56)?;
    let mut model = Model::new(Config::tiny(experts), 56)?;
    // Match the entire shared trunk and expert 0 across the comparison, despite layout offsets.
    for (name, target) in model.parameter_spans() {
        if name.starts_with("router") {
            continue;
        }
        if let Some((_, source)) = dense
            .parameter_spans()
            .into_iter()
            .find(|(n, r)| n == &name && r.len() == target.len())
        {
            model.parameters_mut()[target].copy_from_slice(&dense.parameters()[source]);
        }
    }
    Ok(Trainer::new(model, 5600))
}

fn train(
    trainer: &mut Trainer,
    data: &[u8],
    steps: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if steps == 0 || steps > 100_000 {
        return Err("steps must be 1..=100000; extended training is explicit".into());
    }
    for _ in 0..steps {
        trainer.train_step(data, 16, 0.08)?;
    }
    Ok(())
}

fn demo() -> Result<(), Box<dyn std::error::Error>> {
    let mut dense = initialized(1)?;
    let mut moe = initialized(3)?;
    let dense_before = dense.evaluate(TRAIN)?;
    let moe_before = moe.evaluate(TRAIN)?;
    let parameters_before = moe.model.parameters().to_vec();
    train(&mut dense, TRAIN, 160)?;
    train(&mut moe, TRAIN, 160)?;
    let stats = moe
        .model
        .routing_stats(&TRAIN[..16].iter().map(|&x| x as usize).collect::<Vec<_>>())?;
    println!("same fixture: 160 SGD steps, 16 tokens/step, width 8, expert width 12; matched shared initialization");
    println!(
        "dense parameters: total {}, nonexpert+one-expert {}",
        dense.model.total_parameters(),
        dense.model.active_parameters_per_token()
    );
    println!(
        "MoE parameters: total {}, nonexpert+one-expert {}",
        moe.model.total_parameters(),
        moe.model.active_parameters_per_token()
    );
    println!("uncapped task cross-entropy, mean over ALL corpus targets:");
    println!(
        "dense train loss {dense_before:.4} -> {:.4}; held-out {:.4}",
        dense.evaluate(TRAIN)?,
        dense.evaluate(HELD_OUT)?
    );
    println!(
        "MoE train loss {moe_before:.4} -> {:.4}; held-out {:.4}",
        moe.evaluate(TRAIN)?,
        moe.evaluate(HELD_OUT)?
    );
    println!("capped training probe: routes {:?}; accepted {:?}; dropped {}; capacity {}; mean gate {:.3}", stats.attempted, stats.accepted, stats.dropped, stats.capacity, stats.mean_selected_gate);
    for (name, range) in moe.model.parameter_spans() {
        let change = parameters_before[range.clone()]
            .iter()
            .zip(&moe.model.parameters()[range])
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        println!("parameter change {name}: L2={change:.6}");
    }
    let generated = moe.model.generate(
        &b"rust".iter().map(|&x| x as usize).collect::<Vec<_>>(),
        32,
        &mut Rng::new(7),
        0.8,
    )?;
    println!(
        "sample: {:?}",
        generated
            .into_iter()
            .map(|x| char::from_u32(x as u32).unwrap_or('?'))
            .collect::<String>()
    );
    println!("Tiny overlapping-vocabulary fixture checks mechanics, not language quality, privacy, or scale.");
    Ok(())
}

fn corpus(path: Option<&String>, fallback: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let data = if let Some(path) = path {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(8 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        bytes
    } else {
        fallback.to_vec()
    };
    if data.len() < 2 || data.len() > 8 * 1024 * 1024 || !data.is_ascii() {
        return Err("corpus must be 2 bytes..8 MiB of licensed ASCII text".into());
    }
    Ok(data)
}

fn validate_split(train: &[u8], heldout: &[u8]) -> Result<(), &'static str> {
    if train == heldout {
        return Err("train and held-out text must differ");
    }
    let (short, long) = if train.len() < heldout.len() {
        (train, heldout)
    } else {
        (heldout, train)
    };
    // ponytail: borrowed 32-byte windows use O(short corpus length) memory; use an external dedup index for large corpora.
    let windows: std::collections::HashSet<&[u8]> = short.windows(32).collect();
    if long.windows(32).any(|window| windows.contains(window)) {
        return Err("train and held-out text share a 32-byte passage");
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        None => demo(),
        Some(command @ ("train" | "resume")) if args.len() <= 6 => {
            let path = args.get(2).ok_or("usage: train|resume CHECKPOINT [STEPS] [TRAIN_TEXT] [HELDOUT_TEXT]")?;
            let steps = args.get(3).map_or(Ok(400), |x| x.parse::<usize>())?;
            if args.get(4).is_some() != args.get(5).is_some() { return Err("custom training text requires an explicit held-out text file".into()); }
            let data = corpus(args.get(4), TRAIN)?;
            let heldout = corpus(args.get(5), HELD_OUT)?;
            if args.get(4).is_some() { validate_split(&data, &heldout)?; }
            let mut trainer = if command == "resume" { Trainer::load(Path::new(path))? } else { initialized(3)? };
            train(&mut trainer, &data, steps)?;
            trainer.save(Path::new(path))?;
            println!("saved step {} checkpoint to {path}; train task CE {:.4}; held-out task CE {:.4}", trainer.step, trainer.evaluate(&data)?, trainer.evaluate(&heldout)?);
            Ok(())
        }
        Some("generate") if args.len() <= 5 => {
            let path = args.get(2).ok_or("usage: generate CHECKPOINT PROMPT [TOKENS]")?;
            let prompt = args.get(3).ok_or("usage: generate CHECKPOINT PROMPT [TOKENS]")?;
            if !prompt.is_ascii() || prompt.is_empty() || prompt.len() > 256 { return Err("prompt must be 1..256 ASCII bytes".into()); }
            let count = args.get(4).map_or(Ok(32), |x| x.parse::<usize>())?;
            let trainer = Trainer::load(Path::new(path))?;
            let generated = trainer.model.generate(&prompt.bytes().map(usize::from).collect::<Vec<_>>(), count, &mut Rng::new(7), 0.8)?;
            println!("{}", generated.into_iter().map(|x| char::from_u32(x as u32).unwrap_or('?')).collect::<String>());
            Ok(())
        }
        Some("serve") if args.len() <= 4 => {
            let path = args.get(2).ok_or("usage: serve CHECKPOINT [ADDRESS]")?;
            serve_once(Path::new(path), args.get(3).map_or("127.0.0.1:8787", String::as_str))
        }
        _ => Err("commands: train|resume CHECKPOINT [STEPS] [TRAIN_TEXT] [HELDOUT_TEXT], generate CHECKPOINT PROMPT [TOKENS], serve CHECKPOINT [ADDRESS]".into()),
    }
}

#[test]
fn split_guard_rejects_equal_or_shared_passages() {
    assert!(validate_split(b"same", b"same").is_err());
    assert!(validate_split(
        b"prefix abcdefghijklmnopqrstuvwxyz012345 suffix",
        b"abcdefghijklmnopqrstuvwxyz012345 different"
    )
    .is_err());
    assert!(validate_split(b"training words", b"held out words").is_ok());
}
