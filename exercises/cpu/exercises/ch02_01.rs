// Differentiate Neuron::loss with bias fixed at zero; weight is Neuron::weight.
fn weight_gradient(weight: f64, data: &[(f64, f64)]) -> f64 {
    // TODO: compute the average analytical derivative.
    let _ = (weight, data);
    todo!("average 2 * (weight * input - target) * input")
}
fn main() {
    let _learner_experiment = sim_an_gradient_delta(0.0, 0.0, 2.0, 5.0, 5);
    println!("{}", weight_gradient(0.0, &[(1.0, 2.0)]));
}

// Preserved learner experiment: w/b are the neuron parameters, x/y are input/target.
// This returns one example's contribution to the dataset-mean gradient.
fn sim_an_gradient_delta(w: f64, b: f64, x: f64, y: f64, n: usize) -> (f64, f64) {
    let error = predict(w, b, x) - y;
    (2.0 * error * x / n as f64, 2.0 * error / n as f64)
}

fn predict(w: f64, b: f64, x: f64) -> f64 {
    w * x + b
}

#[test]
fn derivative_uses_every_example() {
    assert!((weight_gradient(0.0, &[(-1.0, -2.0), (1.0, 2.0)]) + 4.0).abs() < 1e-12);
    assert_eq!(weight_gradient(1.0, &[(1.0, 0.0), (2.0, 1.0)]), 3.0);
    assert_eq!(sim_an_gradient_delta(0.0, 0.0, 2.0, 5.0, 5), (-4.0, -2.0));
}
