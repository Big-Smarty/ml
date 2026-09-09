pub fn check() -> Result<(), String> {
    let mut model = Model::new(
        Config {
            vocab_size: 8,
            context: 4,
            width: 3,
            ff_width: 4,
            experts: 2,
            capacity_factor: 10.0,
            balance_weight: 0.0,
        },
        92,
    )
    .map_err(|e| e.to_string())?;
    let bias = model.layout.router_b.start;
    model.parameters[bias + 1] += 1.0;
    let input = [1, 3, 5];
    let target = [3, 5, 2];
    let g = model
        .loss_and_gradient(&input, &target)
        .map_err(|e| e.to_string())?;
    crate::ensure(g.routes.iter().all(|&e|e==1),&format!("learned routing must follow expert-one bias; got {:?}, indicating the fixed dense baseline",g.routes))?;
    crate::ensure(
        g.values[model.layout.router_w.clone()]
            .iter()
            .any(|x| x.abs() > 1e-10),
        "task-only router gradient is zero",
    )?;
    for i in 0..model.parameters.len() {
        let old = model.parameters[i];
        model.parameters[i] = old + 1e-5;
        let plus = model.loss(&input, &target).map_err(|e| e.to_string())?;
        model.parameters[i] = old - 1e-5;
        let minus = model.loss(&input, &target).map_err(|e| e.to_string())?;
        model.parameters[i] = old;
        let numeric = (plus - minus) / 2e-5;
        crate::ensure(
            (numeric - g.values[i]).abs() <= 1e-6 + 1e-4 * numeric.abs().max(g.values[i].abs()),
            &format!(
                "MoE/backbone derivative {i}: actual {}, numerical {numeric}",
                g.values[i]
            ),
        )?;
    }
    model.config.capacity_factor = 0.1;
    let stats = model
        .routing_stats(&[1, 3, 5, 7])
        .map_err(|e| e.to_string())?;
    crate::ensure(
        stats.capacity == 1 && stats.dropped == 3,
        "capacity must keep one token and drop three into the residual path",
    )?;
    let full = model.forward(&[1, 3, 5, 7]).map_err(|e| e.to_string())?;
    let prefix = model.forward(&[1, 3]).map_err(|e| e.to_string())?;
    crate::ensure(
        full[..prefix.len()] == prefix,
        "inference must be prefix invariant despite restrictive training capacity",
    )?;
    let mut trainer = initialized(3).map_err(|e| e.to_string())?;
    let before = trainer.evaluate(TRAIN).map_err(|e| e.to_string())?;
    train(&mut trainer, TRAIN, 160).map_err(|e| e.to_string())?;
    let after = trainer.evaluate(TRAIN).map_err(|e| e.to_string())?;
    crate::ensure(
        after < before * 0.5,
        &format!("actual sparse LM training failed: {before} -> {after}"),
    )?;
    let heldout = trainer.evaluate(HELD_OUT).map_err(|e| e.to_string())?;
    println!("actual sparse LM: train CE {before:.6} -> {after:.6}; held-out CE {heldout:.6}; 2560 token presentations");
    Ok(())
}

pub fn run(args: &[String]) -> Result<(), String> {
    command(args).map_err(|e| e.to_string())
}
