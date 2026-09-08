//! Selective scalar state-space recurrence, associative scan, and causal linear attention.
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
struct Transition {
    a: f64,
    b: f64,
}

impl Transition {
    /// Apply `earlier`, then `self`.
    fn after(self, earlier: Self) -> Self {
        Self {
            a: self.a * earlier.a,
            b: self.a * earlier.b + self.b,
        }
    }

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

fn associative_scan(transitions: &[Transition], initial: f64) -> Vec<f64> {
    // ponytail: O(N log N) sequential work; use a work-efficient parallel scan for throughput.
    let mut prefixes = transitions.to_vec();
    let mut offset = 1;
    while offset < prefixes.len() {
        let previous = prefixes.clone();
        for i in offset..prefixes.len() {
            prefixes[i] = previous[i].after(previous[i - offset]);
        }
        offset *= 2;
    }
    prefixes
        .into_iter()
        .map(|transition| transition.apply(initial))
        .collect()
}

fn feature(value: [f64; 2]) -> [f64; 2] {
    value.map(|component| component.max(0.0) + 1.0)
}

fn causal_linear_attention_recurrent(
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
    let mut summary = [0.0; 2];
    let mut normalizer = [0.0; 2];
    let mut output = Vec::with_capacity(values.len());
    for ((&query, &key), &value) in queries.iter().zip(keys).zip(values) {
        let key_feature = feature(key);
        for d in 0..2 {
            summary[d] += key_feature[d] * value;
            normalizer[d] += key_feature[d];
        }
        let query_feature = feature(query);
        let numerator: f64 = query_feature.iter().zip(summary).map(|(q, s)| q * s).sum();
        let denominator: f64 = query_feature
            .iter()
            .zip(normalizer)
            .map(|(q, z)| q * z)
            .sum();
        let value = numerator / (denominator + 1e-12);
        if !value.is_finite() {
            return Err("linear attention output became nonfinite");
        }
        output.push(value);
    }
    Ok(output)
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        return Err("recurrent and direct linear attention disagree".into());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn associative_scan_matches_recurrence() -> Result<(), &'static str> {
        let transitions = selective_transitions(&[0.2, -0.5, 1.0, 0.1, -0.3])?;
        assert!(close(
            &recurrent(&transitions, 0.4),
            &associative_scan(&transitions, 0.4)
        ));
        Ok(())
    }

    #[test]
    fn transition_composition_is_associative() {
        let a = Transition { a: 0.2, b: 1.0 };
        let b = Transition { a: -0.3, b: 0.4 };
        let c = Transition { a: 0.8, b: -0.2 };
        let left = c.after(b).after(a);
        let right = c.after(b.after(a));
        assert!((left.a - right.a).abs() < 1e-12 && (left.b - right.b).abs() < 1e-12);
    }

    #[test]
    fn linear_attention_recurrence_matches_scalar_oracle() -> Result<(), &'static str> {
        let queries = [[0.2, -0.1], [0.5, 0.3], [-0.2, 0.7]];
        let keys = [[0.4, 0.1], [-0.1, 0.2], [0.3, -0.5]];
        let values = [2.0, -1.0, 0.5];
        assert!(close(
            &causal_linear_attention_recurrent(&queries, &keys, &values)?,
            &causal_linear_attention_scalar(&queries, &keys, &values)?
        ));
        Ok(())
    }

    #[test]
    fn invalid_sequence_shapes_are_rejected() {
        assert!(selective_transitions(&[]).is_err());
        assert!(selective_transitions(&[f64::NAN]).is_err());
        assert!(selective_transitions(&[f64::MAX]).is_err());
        for invalid in [f64::NAN, f64::INFINITY] {
            assert!(
                causal_linear_attention_recurrent(&[[invalid, 0.0]], &[[0.0; 2]], &[1.0]).is_err()
            );
            assert!(causal_linear_attention_scalar(&[[0.0; 2]], &[[0.0; 2]], &[invalid]).is_err());
        }
        assert!(causal_linear_attention_recurrent(&[[0.0, 0.0]], &[], &[]).is_err());
    }
}
