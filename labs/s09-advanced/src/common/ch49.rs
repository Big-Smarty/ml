// Selective scalar state-space recurrence, associative scan, and causal linear attention.
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
struct Transition {
    a: f64,
    b: f64,
}

impl Transition {
    fn apply(self, state: f64) -> f64 {
        self.a * state + self.b
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn selective_transitions(inputs: &[f64]) -> Result<Vec<Transition>, &'static str> {
    if inputs.is_empty() || inputs.iter().any(|value| !value.is_finite()) {
        return Err("sequence must contain finite values");
    }
    let transitions: Vec<_> = inputs
        .iter()
        .map(|&input| Transition {
            a: sigmoid(1.2 - 0.8 * input.abs()),
            b: (0.7 + 0.2 * input) * input,
        })
        .collect();
    if transitions
        .iter()
        .any(|transition| !transition.b.is_finite())
    {
        return Err("selective input contribution overflowed");
    }
    Ok(transitions)
}

fn recurrent(transitions: &[Transition], initial: f64) -> Vec<f64> {
    let mut state = initial;
    transitions
        .iter()
        .map(|&transition| {
            state = transition.apply(state);
            state
        })
        .collect()
}

fn feature(value: [f64; 2]) -> [f64; 2] {
    value.map(|component| component.max(0.0) + 1.0)
}

fn causal_linear_attention_scalar(
    queries: &[[f64; 2]],
    keys: &[[f64; 2]],
    values: &[f64],
) -> Result<Vec<f64>, &'static str> {
    if queries.is_empty() || queries.len() != keys.len() || keys.len() != values.len() {
        return Err("queries, keys, and values need the same nonzero length");
    }
    if queries
        .iter()
        .chain(keys)
        .flatten()
        .chain(values)
        .any(|value| !value.is_finite())
    {
        return Err("linear attention inputs must be finite");
    }
    let output: Vec<_> = queries
        .iter()
        .enumerate()
        .map(|(time, &query)| {
            let query_feature = feature(query);
            let (numerator, denominator) = (0..=time).fold((0.0, 0.0), |acc, past| {
                let key_feature = feature(keys[past]);
                let similarity: f64 = query_feature
                    .iter()
                    .zip(key_feature)
                    .map(|(q, k)| q * k)
                    .sum();
                (acc.0 + similarity * values[past], acc.1 + similarity)
            });
            numerator / (denominator + 1e-12)
        })
        .collect();
    if output.iter().any(|value| !value.is_finite()) {
        return Err("direct attention output became nonfinite");
    }
    Ok(output)
}

fn close(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(a, b)| {
            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 + 1e-8 * a.abs().max(b.abs())
        })
}

fn demo() -> Result<(), Box<dyn std::error::Error>> {
    let inputs: Vec<_> = (0..4_096).map(|i| (i as f64 * 0.013).sin()).collect();
    let transitions = selective_transitions(&inputs)?;
    let start = Instant::now();
    let recurrent_output = black_box(recurrent(&transitions, 0.25));
    let recurrent_time = start.elapsed();
    let start = Instant::now();
    let scan_output = black_box(associative_scan(&transitions, 0.25));
    let scan_time = start.elapsed();
    if !close(&recurrent_output, &scan_output) {
        return Err("scan and recurrence disagree".into());
    }

    let queries: Vec<_> = inputs.iter().map(|&x| [x, 0.5 * x]).collect();
    let keys: Vec<_> = inputs.iter().map(|&x| [0.25 * x, -x]).collect();
    let start = Instant::now();
    let attention = black_box(causal_linear_attention_recurrent(&queries, &keys, &inputs)?);
    let attention_time = start.elapsed();
    let prefix = 256;
    let start = Instant::now();
    let scalar = black_box(causal_linear_attention_scalar(
        &queries[..prefix],
        &keys[..prefix],
        &inputs[..prefix],
    )?);
    let scalar_time = start.elapsed();
    if !close(&attention[..prefix], &scalar) {
        println!("learning goal: recurrent result differs from feature-kernel oracle");
    }
    println!(
        "length={}, release={}",
        inputs.len(),
        !cfg!(debug_assertions)
    );
    println!(
        "SSM recurrence={recurrent_time:?}, sequential scan schedule={scan_time:?}, final={:.6}",
        recurrent_output[recurrent_output.len() - 1]
    );
    println!(
        "linear-attention recurrence={attention_time:?}, final={:.6}",
        attention[attention.len() - 1]
    );
    println!("direct scalar oracle for first {prefix} tokens={scalar_time:?}");
    println!("Fixed selective coefficients are illustrative, not a trained Mamba layer.");
    println!("The scan schedule exposes parallel structure but this scalar program executes it sequentially.");
    Ok(())
}
