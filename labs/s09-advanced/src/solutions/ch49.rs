//! Worked solution: Affine composition is associative but ordered. Each doubling round reads a clone of the entire preceding round, avoiding in-place dependency corruption. Feature attention accumulates key-value and key-only summaries before the inclusive query read. Numerical parity proves the kernel identity; author source review verifies the scan schedule.
//! Compare intermediate quantities with the lesson before using the final goal check.
//! Worked solution for chapter 49. Read the comments and lesson explanations before comparing.
include!("../common/ch49.rs");
include!("../checks/ch49.rs");
impl Transition {
    fn after(self, earlier: Self) -> Self {
        Self {
            a: self.a * earlier.a,
            b: self.a * earlier.b + self.b,
        }
    }
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
