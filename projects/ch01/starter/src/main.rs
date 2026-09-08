// Complete the prediction rule, then replace numerical slopes with your own experiment.
#[cfg(test)]
fn predict(weight: f64, bias: f64, input: f64) -> f64 {
    weight * input + bias
}
fn main() {
    let samples = [
        (-2.0, -3.0),
        (-1.0, -1.0),
        (0.0, 1.0),
        (1.0, 3.0),
        (2.0, 5.0),
    ];
    for (input, target) in samples {
        println!("input={input:>4}, target={target:>4}");
    }
    println!("Five observations, ready for your predictor. Complete predict, then run cargo test.");

    let _ = train(&samples, 1000, 0.00001, 0.1);
}

fn train(samples: &[(f64, f64)], iterations: usize, h: f64, eta: f64) -> (f64, f64) {
    let mut w = 0.0;
    let mut b = 0.0;
    let (input, expected): (Vec<_>, Vec<_>) = samples.iter().map(|s| *s).unzip();
    for _ in 0..iterations {
        println!("w: {w}, b: {b}");
        println!("loss: {}", mse(w, b, &input, &expected));
        w = new_w(w, b, &input, &expected, h, eta);
        b = new_b(w, b, &input, &expected, h, eta)
    }
    println!("w: {w}, b: {b}");
    println!("loss: {}", mse(w, b, &input, &expected));
    (w, b)
}

fn mse(w: f64, b: f64, input: &Vec<f64>, expected: &Vec<f64>) -> f64 {
    input
        .iter()
        .zip(expected.iter())
        .map(|(x, y)| (w * x + b - y).powi(2))
        .sum::<f64>()
        / (input.len() as f64)
}

fn gradient_weight(w: f64, b: f64, input: &Vec<f64>, expected: &Vec<f64>, h: f64) -> f64 {
    (mse(w + h, b, input, expected) - mse(w - h, b, input, expected)) / (2.0 * h)
}

fn gradient_bias(w: f64, b: f64, input: &Vec<f64>, expected: &Vec<f64>, h: f64) -> f64 {
    (mse(w, b + h, input, expected) - mse(w, b - h, input, expected)) / (2.0 * h)
}

fn new_w(w: f64, b: f64, input: &Vec<f64>, expected: &Vec<f64>, h: f64, eta: f64) -> f64 {
    w - eta * gradient_weight(w, b, input, expected, h)
}

fn new_b(w: f64, b: f64, input: &Vec<f64>, expected: &Vec<f64>, h: f64, eta: f64) -> f64 {
    b - eta * gradient_bias(w, b, input, expected, h)
}

#[test]
fn held_out_input() {
    assert!((predict(2.0, 1.0, 0.5) - 2.0).abs() < 1e-12);
    assert_eq!(predict(-1.0, 4.0, 3.0), 1.0);
}
