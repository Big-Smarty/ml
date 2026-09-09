pub fn check() -> Result<(), String> {
    let ts = [
        Transition { a: 0.5, b: 1.0 },
        Transition { a: 0.2, b: -0.5 },
        Transition { a: 0.8, b: 0.1 },
    ];
    crate::ensure(
        close(&associative_scan(&ts, 1.0), &[1.5, -0.2, -0.06]),
        "scan must compose ordered transitions from a nonzero initial state",
    )?;
    let selective = selective_transitions(&[0.2, -0.5, 1.0, 0.1, -0.3])?;
    crate::ensure(
        close(
            &associative_scan(&selective, 0.4),
            &recurrent(&selective, 0.4),
        ),
        "five-position selective scan differs from recurrence",
    )?;
    let q = [[0.2, -0.1], [0.5, 0.3], [-0.2, 0.7]];
    let k = [[0.4, 0.1], [-0.1, 0.2], [0.3, -0.5]];
    let v = [2.0, -1.0, 0.5];
    let actual = causal_linear_attention_recurrent(&q, &k, &v)?;
    let expected = causal_linear_attention_scalar(&q, &k, &v)?;
    crate::ensure(close(&actual,&expected), &format!("feature-kernel summaries: actual {actual:?}, direct oracle {expected:?}; uniform averaging loses query/key weights"))?;
    let changed = [2.0, -1.0, 99.0];
    let future = causal_linear_attention_recurrent(&q, &k, &changed)?;
    crate::ensure(
        close(&actual[..2], &future[..2]),
        "future values leaked into earlier outputs",
    )?;
    crate::ensure(
        causal_linear_attention_recurrent(&q, &[], &v).is_err(),
        "unequal sequence shapes accepted",
    )
}

pub fn run(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("this experiment takes no extra arguments".into());
    }
    demo().map_err(|e| e.to_string())
}
