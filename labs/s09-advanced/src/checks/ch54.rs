pub fn check() -> Result<(), String> {
    let model = Model {
        weights: [0.8, 0.4],
        bias: -0.2,
        threshold: 0.5,
    };
    let a = group_metrics(&model, &ROWS, 'A')?;
    let b = group_metrics(&model, &ROWS, 'B')?;
    crate::ensure(a.count==6 && b.count==6 && a.true_positive_rate==Some(2.0/3.0) && b.true_positive_rate==Some(0.5),&format!("group denominators: A={a:?}, B={b:?}; pooled accuracy cannot substitute for group rates"))?;
    let negatives = [Row {
        features: [0.0; 2],
        group: 'C',
        label: false,
        member: false,
    }];
    crate::ensure(
        group_metrics(&model, &negatives, 'C')?
            .true_positive_rate
            .is_none(),
        "no positive labels means undefined TPR",
    )?;
    let attr = local_attribution(&model, [1.0, 1.0], [0.0; 2])?;
    crate::ensure(
        crate::close(attr[0], 0.8) && crate::close(attr[1], 0.4),
        "attribution must compare with the declared baseline",
    )?;
    let randomized = Model {
        weights: [-0.3, 0.9],
        ..model
    };
    crate::ensure(
        local_attribution(&randomized, [1.0, 1.0], [0.0; 2])? != attr,
        "attribution ignored model randomization",
    )?;
    crate::ensure(
        robustness(&model, &ROWS, 0.25)?.0 == 4,
        "named endpoint perturbation must count four flips",
    )?;
    let trained = train_on_members(&ROWS)?;
    let attack = membership_attack(&trained, &ROWS, 0.45)?;
    crate::ensure(
        crate::close(
            attack.advantage,
            attack.true_positive_rate - attack.false_positive_rate,
        ),
        "membership advantage requires separate populations",
    )?;
    let strata = [
        Stratum {
            treated_total: 10,
            treated_outcomes: 1,
            control_total: 90,
            control_outcomes: 18,
        },
        Stratum {
            treated_total: 90,
            treated_outcomes: 54,
            control_total: 10,
            control_outcomes: 7,
        },
    ];
    let (crude, standard) = association_contrasts(&strata)?;
    crate::ensure(
        crate::close(crude, 0.3) && crate::close(standard, -0.1),
        "standardization must expose association reversal",
    )?;
    crate::ensure(
        parse_input("schema=1,f0=inf,f1=0").is_err(),
        "security boundary accepted infinity",
    )
}

pub fn run(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("this experiment takes no extra arguments".into());
    }
    demo().map_err(|e| e.to_string())
}
