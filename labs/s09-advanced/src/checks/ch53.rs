pub fn check() -> Result<(), String> {
    let previous = train(&TRAIN, 1)?;
    let candidate = train(&TRAIN, 2)?;
    crate::ensure(choose_release(&candidate,&previous,&[8.5,10.0,11.5],0.01)?.model_version==2,"a numerically equivalent candidate should pass the canary, not be rejected unconditionally")?;
    let mut bad = candidate.clone();
    bad.scale *= 2.0;
    crate::ensure(
        choose_release(&bad, &previous, &[8.0, 12.0], 0.01)?.model_version == 1,
        "preprocessing skew must trigger rollback",
    )?;
    let mut monitor = Monitor::default();
    for _ in 0..20 {
        monitor.observe(10.0)?;
    }
    for x in [12.0, 12.0, 12.0, 12.0] {
        monitor.observe(x)?;
    }
    crate::ensure(
        monitor.mean() == Some(12.0) && monitor.drift_from(10.0, 1.0)?,
        "four-request rolling window should detect recent shift instead of diluting it in history",
    )?;
    crate::ensure(
        Model::decode(&candidate.encode())? == candidate
            && crate::close(candidate.predict(12.0)?, 3.0),
        "artifact round trip or train/serve parity failed",
    )?;
    crate::ensure(
        Model::decode("MLMODEL 2 1 1 0 1 1 0 0").is_err() && parse_body("schema=1&x=NaN").is_err(),
        "bad format or nonfinite request accepted",
    )
}

pub fn run(args: &[String]) -> Result<(), String> {
    command(args).map_err(|e| e.to_string())
}
