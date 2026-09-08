#[derive(Debug)]
struct Attention {
    output: Vec<f64>,
    probabilities: Vec<f64>,
}
#[derive(Debug)]
struct AttentionGradients {
    query_gradients: Vec<f64>,
    key_gradients: Vec<f64>,
    value_gradients: Vec<f64>,
}
fn causal_attention(
    queries: &[f64],
    keys: &[f64],
    values: &[f64],
    positions: usize,
    width: usize,
) -> Attention {
    assert!(positions > 0 && width > 0);
    let len = positions
        .checked_mul(width)
        .expect("attention shape overflow");
    assert_eq!(queries.len(), len);
    assert_eq!(keys.len(), len);
    assert_eq!(values.len(), len);
    let mut probabilities = vec![0.0; positions * positions];
    let mut output = vec![0.0; len];
    let scale = (width as f64).sqrt().recip();
    for i in 0..positions {
        let mut max = f64::NEG_INFINITY;
        for j in 0..=i {
            let score = (0..width)
                .map(|z| queries[i * width + z] * keys[j * width + z])
                .sum::<f64>()
                * scale;
            probabilities[i * positions + j] = score;
            max = max.max(score)
        }
        let mut sum = 0.0;
        for j in 0..=i {
            probabilities[i * positions + j] = (probabilities[i * positions + j] - max).exp();
            sum += probabilities[i * positions + j]
        }
        for j in 0..=i {
            probabilities[i * positions + j] /= sum;
            for z in 0..width {
                output[i * width + z] += probabilities[i * positions + j] * values[j * width + z]
            }
        }
    }
    Attention {
        output,
        probabilities,
    }
}
fn backward(
    queries: &[f64],
    keys: &[f64],
    values: &[f64],
    probabilities: &[f64],
    output_gradients: &[f64],
    positions: usize,
    width: usize,
) -> AttentionGradients {
    assert!(positions > 0 && width > 0);
    let len = positions
        .checked_mul(width)
        .expect("attention shape overflow");
    assert_eq!(queries.len(), len);
    assert_eq!(keys.len(), len);
    assert_eq!(values.len(), len);
    assert_eq!(output_gradients.len(), len);
    assert_eq!(
        probabilities.len(),
        positions
            .checked_mul(positions)
            .expect("probability shape overflow")
    );
    let mut query_gradients = vec![0.0; len];
    let mut key_gradients = vec![0.0; len];
    let mut value_gradients = vec![0.0; len];
    let scale = (width as f64).sqrt().recip();
    for i in 0..positions {
        let mut probability_gradients = vec![0.0; i + 1];
        for j in 0..=i {
            for z in 0..width {
                probability_gradients[j] += output_gradients[i * width + z] * values[j * width + z];
                value_gradients[j * width + z] +=
                    probabilities[i * positions + j] * output_gradients[i * width + z]
            }
        }
        let dot = (0..=i)
            .map(|j| probability_gradients[j] * probabilities[i * positions + j])
            .sum::<f64>();
        for j in 0..=i {
            let dot_product_gradient =
                probabilities[i * positions + j] * (probability_gradients[j] - dot) * scale;
            for z in 0..width {
                query_gradients[i * width + z] += dot_product_gradient * keys[j * width + z];
                key_gradients[j * width + z] += dot_product_gradient * queries[i * width + z]
            }
        }
    }
    AttentionGradients {
        query_gradients,
        key_gradients,
        value_gradients,
    }
}
#[cfg(test)]
fn attention_probe_objective(
    queries: &[f64],
    keys: &[f64],
    values: &[f64],
    positions: usize,
    width: usize,
    output_gradients: &[f64],
) -> f64 {
    let output = causal_attention(queries, keys, values, positions, width).output;
    assert_eq!(output.len(), output_gradients.len());
    output
        .iter()
        .zip(output_gradients)
        .map(|(value, gradient)| value * gradient)
        .sum()
}
fn main() {
    let q = [1.0, 0.0, 0.0, 1.0];
    let k = [1.0, 0.0, 1.0, 1.0];
    let v = [2.0, 0.0, 0.0, 4.0];
    let attention = causal_attention(&q, &k, &v, 2, 2);
    println!(
        "causal probabilities [q0->k0,q0->k1,q1->k0,q1->k1] = {:.3?}",
        attention.probabilities
    );
    println!("outputs = {:.3?}", attention.output);
    let gradients = backward(&q, &k, &v, &attention.probabilities, &[1.0; 4], 2, 2);
    println!(
        "gradient norms: Q={:.4} K={:.4} V={:.4}",
        norm(&gradients.query_gradients),
        norm(&gradients.key_gradients),
        norm(&gradients.value_gradients)
    );
}
fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn future_is_exactly_masked() {
        let q = [1.0, 0.5, 0.2, -0.4, 0.3, 0.7];
        let k = [0.2, 0.1, 0.4, 0.3, 9.0, 9.0];
        let v = [1.0, 2.0, 3.0, 4.0, 99.0, 99.0];
        let attention = causal_attention(&q, &k, &v, 3, 2);
        assert_eq!(&attention.output[..2], &v[..2]);
        assert_eq!(attention.probabilities[1], 0.0);
        assert_eq!(attention.probabilities[2], 0.0);
        for row in attention.probabilities.chunks_exact(3) {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        let mut changed_k = k;
        changed_k[4] = -500.0;
        changed_k[5] = 700.0;
        let mut changed_v = v;
        changed_v[4] = -900.0;
        changed_v[5] = 600.0;
        let changed = causal_attention(&q, &changed_k, &changed_v, 3, 2);
        assert_eq!(&attention.output[..4], &changed.output[..4]);
    }
    fn central_difference(
        queries: &[f64],
        keys: &[f64],
        values: &[f64],
        input_group: usize,
        index: usize,
        output_gradients: &[f64],
    ) -> f64 {
        let eps = 1e-5;
        let (mut query_plus, mut key_plus, mut value_plus) =
            (queries.to_vec(), keys.to_vec(), values.to_vec());
        [&mut query_plus, &mut key_plus, &mut value_plus][input_group][index] += eps;
        let plus =
            attention_probe_objective(&query_plus, &key_plus, &value_plus, 3, 2, output_gradients);
        [&mut query_plus, &mut key_plus, &mut value_plus][input_group][index] -= 2.0 * eps;
        let minus =
            attention_probe_objective(&query_plus, &key_plus, &value_plus, 3, 2, output_gradients);
        (plus - minus) / (2.0 * eps)
    }
    #[test]
    fn qkv_gradients_match_central_differences() {
        let q = vec![0.7, -0.2, 0.1, 0.9, -0.5, 0.3];
        let k = vec![0.4, 0.6, -0.8, 0.2, 0.3, -0.7];
        let v = vec![1.0, -0.5, 0.2, 0.8, -0.4, 0.9];
        let up = [0.3, -0.7, 1.2, 0.4, -0.2, 0.6];
        let attention = causal_attention(&q, &k, &v, 3, 2);
        let gradients = backward(&q, &k, &v, &attention.probabilities, &up, 3, 2);
        for (group, analytic) in [
            &gradients.query_gradients,
            &gradients.key_gradients,
            &gradients.value_gradients,
        ]
        .iter()
        .enumerate()
        {
            for (i, &analytic) in analytic.iter().enumerate() {
                let numerical = central_difference(&q, &k, &v, group, i, &up);
                assert!((analytic - numerical).abs() < 1e-6 + 1e-4 * numerical.abs());
            }
        }

        let doubled_output_gradients = up.map(|gradient| 2.0 * gradient);
        let doubled = backward(
            &q,
            &k,
            &v,
            &attention.probabilities,
            &doubled_output_gradients,
            3,
            2,
        );
        for (original, doubled) in [
            &gradients.query_gradients,
            &gradients.key_gradients,
            &gradients.value_gradients,
        ]
        .iter()
        .zip([
            &doubled.query_gradients,
            &doubled.key_gradients,
            &doubled.value_gradients,
        ]) {
            for (original, doubled) in original.iter().zip(doubled) {
                assert!((2.0 * original - doubled).abs() < 1e-12);
            }
        }
    }
}
