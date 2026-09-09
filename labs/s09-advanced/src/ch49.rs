//! Chapter 49 learner algorithms. Baseline computes a serial recurrence and uniform causal average. Build ordered affine scan and feature-kernel recurrent summaries; the feature-kernel output must match its direct oracle.
include!("common/ch49.rs");
include!("checks/ch49.rs");
impl Transition {
    fn after(self, earlier: Self) -> Self {
        Self {
            a: self.a * earlier.a,
            b: self.a * earlier.b + self.b,
        }
    }
}

fn associative_scan(transitions: &[Transition], initial: f64) -> Vec<f64> {
    // Baseline: serial prefix composition; replace with offset-doubling rounds.
    let mut prefix = Transition { a: 1.0, b: 0.0 };
    transitions
        .iter()
        .map(|&next| {
            prefix = next.after(prefix);
            prefix.apply(initial)
        })
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
    // Uniform causal averaging is a working attention baseline.
    let mut sum = 0.0;
    Ok(values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            sum += v;
            sum / (i + 1) as f64
        })
        .collect())
}
