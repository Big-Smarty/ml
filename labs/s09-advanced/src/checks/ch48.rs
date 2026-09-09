pub fn check() -> Result<(), String> {
    let initial = Model::new();
    let example = [([1.0, 0.2], 1.4)];
    let selected = initial.route(example[0].0).0;
    let mut model = initial.clone();
    model.train_epoch(&example, 1e-5, 3.0, 0.0, 0)?;
    let analytic = (initial.router_weights[selected][0] - model.router_weights[selected][0]) / 1e-5;
    let mut plus = initial.clone();
    let mut minus = initial.clone();
    plus.router_weights[selected][0] += 1e-5;
    minus.router_weights[selected][0] -= 1e-5;
    let numerical = (plus.loss(&example)? - minus.loss(&example)?) / 2e-5;
    crate::ensure(numerical.abs()>1e-8 && (analytic-numerical).abs()<1e-6+1e-4*numerical.abs(), &format!("router task derivative: actual {analytic:.8}, central difference {numerical:.8}; frozen or normalized-to-one gates cannot learn"))?;
    let repeated = [([1.0, 0.0], 1.4); 4];
    let mut cap = Model::new();
    crate::ensure(
        cap.train_epoch(&repeated, 0.01, 0.1, 0.02, 0)? == 3,
        "capacity one must drop three of four identical routes",
    )?;
    let mut balanced = initial.clone();
    let mut task = initial.clone();
    balanced.train_epoch(&example, 0.01, 3.0, 0.3, 0)?;
    task.train_epoch(&example, 0.01, 3.0, 0.0, 0)?;
    crate::ensure(
        balanced.router_weights != task.router_weights,
        "auxiliary balance must affect the router",
    )?;
    let (trained, before, after, _) = train_fixture(1500)?;
    crate::ensure(
        after < before * 0.2,
        &format!("joint fit insufficient: {before} -> {after}"),
    )?;
    let mut loads = [0; EXPERTS];
    for &(x, _) in &fixture() {
        loads[trained.route(x).0] += 1;
    }
    crate::ensure(
        loads.iter().all(|&n| n > 0),
        "an expert never received a route",
    )
}

pub fn run(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("this experiment takes no extra arguments".into());
    }
    demo().map_err(|e| e.to_string())
}
