use std::hint::black_box;

fn dense(
    inputs: &[f32],
    weights: &[f32],
    bias: &[f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
    outputs: &mut [f32],
) {
    for b in 0..batch {
        for o in 0..out_features {
            let mut sum = bias[o];
            for i in 0..in_features {
                sum += inputs[b * in_features + i] * weights[o * in_features + i];
            }
            outputs[b * out_features + o] = sum;
        }
    }
}

#[cfg(test)]
fn median_ns(_samples: &mut [u128]) -> u128 {
    // TODO: sort measured durations and return their middle value.
    todo!("complete the benchmark statistic")
}

fn main() {
    let inputs = [1.0, 2.0, 3.0, 4.0];
    let weights = [0.5, -1.0];
    let bias = [0.0];
    let mut outputs = [0.0; 2];
    dense(
        black_box(&inputs),
        black_box(&weights),
        black_box(&bias),
        2,
        2,
        1,
        black_box(&mut outputs),
    );
    println!("prior dense inference: {outputs:?}; now add warmups and repeated timing in the test");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reports_median_not_best_case() {
        let mut s = [90, 12, 31, 30, 29];
        assert_eq!(median_ns(&mut s), 30);
    }
}
