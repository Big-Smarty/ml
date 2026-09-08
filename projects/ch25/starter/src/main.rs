use std::hint::black_box;

fn dense(x: &[f32], w: &[f32], batch: usize, input: usize, output: usize, y: &mut [f32]) {
    for b in 0..batch {
        for o in 0..output {
            let mut sum = 0.0;
            for i in 0..input {
                sum += x[b * input + i] * w[o * input + i];
            }
            y[b * output + o] = sum;
        }
    }
}

#[cfg(test)]
fn median_ns(_samples: &mut [u128]) -> u128 {
    // TODO: sort measured durations and return their middle value.
    todo!("complete the benchmark statistic")
}

fn main() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let w = [0.5, -1.0];
    let mut y = [0.0; 2];
    dense(black_box(&x), black_box(&w), 2, 2, 1, black_box(&mut y));
    println!("prior dense inference: {y:?}; now add warmups and repeated timing in the test");
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
