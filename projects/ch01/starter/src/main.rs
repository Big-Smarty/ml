// Complete the prediction rule, then replace numerical slopes with your own experiment.
#[cfg(test)]
fn predict(weight: f64, bias: f64, input: f64) -> f64 {
    let _ = (weight, bias, input);
    todo!("Multiply the input by the weight, then add the bias")
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
}
#[test]
fn held_out_input() {
    assert!((predict(2.0, 1.0, 0.5) - 2.0).abs() < 1e-12);
    assert_eq!(predict(-1.0, 4.0, 3.0), 1.0);
}
