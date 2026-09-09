//! Learner: replace frozen-feature training with all nine mean derivatives.
use crate::xor::*;

pub fn gradient(net: &Net, data: &[Example]) -> Result<Gradient, &'static str> {
    validate_data(data)?;
    let mut gradient = Gradient::default();
    // Working baseline: logistic regression on two fixed sigmoid features.
    // Learn the output layer only; hidden parameters stay at their initial values.
    for &(features, target) in data {
        let forward = net.forward(features);
        let signal = (forward.probability - target) / data.len() as f64;
        for (g, h) in gradient.output_weights.iter_mut().zip(forward.hidden) {
            *g += signal * h;
        }
        gradient.output_bias += signal;
    }
    Ok(gradient)
}
pub fn run(args: &[String]) -> Result<(), String> {
    report(args, gradient)
}
pub fn check() -> Result<(), String> {
    verify(gradient)
}
pub fn report(args: &[String], derivative: Derivative) -> Result<(), String> {
    let (mut steps, mut rate, mut shift) = (10_000usize, 1.0f64, 0.0f64);
    let mut pairs = args.chunks_exact(2);
    for pair in &mut pairs {
        match pair[0].as_str() {
            "--steps" => steps = pair[1].parse().map_err(|_| "steps must be an integer")?,
            "--rate" => rate = pair[1].parse().map_err(|_| "rate must be a number")?,
            "--shift" => shift = pair[1].parse().map_err(|_| "shift must be a number")?,
            _ => return Err("XOR options: --steps N --rate R --shift S".into()),
        }
    }
    if !pairs.remainder().is_empty() || steps > 100_000 || !shift.is_finite() || shift.abs() > 5. {
        return Err("XOR options require values; steps <=100000 and finite |shift|<=5".into());
    }
    validate_learning_rate(rate)?;
    let mut initial = Net::new();
    initial.hidden_weights[0][0] += shift;
    println!("steps={steps}, learning rate={rate}, first hidden weight shift={shift}");
    println!(
        "XOR: 00→0, 01→1, 10→1, 11→0. Constant p=0.5 mean BCE={:.6}",
        2f64.ln()
    );
    let model = initial.train(&XOR, steps, rate, derivative)?;
    println!(
        "mean BCE {:.6} -> {:.6}",
        initial.loss(&XOR)?,
        model.loss(&XOR)?
    );
    for &(x, y) in &XOR {
        println!(
            "x={x:?} target={y} hidden={:?} p={:.5} class={}",
            model.forward(x).hidden,
            model.probability(x),
            model.predict(x)
        );
    }
    println!(
        "unseen [0.1,0.9]: p={:.5}; no noisy-input correctness promise",
        model.probability([0.1, 0.9])
    );
    Ok(())
}
fn parameter(net: &mut Net, index: usize) -> &mut f64 {
    match index {
        0..=3 => &mut net.hidden_weights[index / 2][index % 2],
        4..=5 => &mut net.hidden_bias[index - 4],
        6..=7 => &mut net.output_weights[index - 6],
        _ => &mut net.output_bias,
    }
}
fn components(g: Gradient) -> [f64; 9] {
    [
        g.hidden_weights[0][0],
        g.hidden_weights[0][1],
        g.hidden_weights[1][0],
        g.hidden_weights[1][1],
        g.hidden_bias[0],
        g.hidden_bias[1],
        g.output_weights[0],
        g.output_weights[1],
        g.output_bias,
    ]
}
pub fn verify(derivative: Derivative) -> Result<(), String> {
    let alternate = [([0.2, -0.7], 0.3), ([1.1, 0.4], 1.), ([-0.5, 0.8], 0.)];
    for data in [&XOR[..], &alternate[..]] {
        let model = Net::new();
        let actual = components(derivative(&model, data)?);
        for (j, &g) in actual.iter().enumerate() {
            let (mut plus, mut minus) = (model, model);
            *parameter(&mut plus, j) += 1e-5;
            *parameter(&mut minus, j) -= 1e-5;
            let numeric = (plus.loss(data)? - minus.loss(data)?) / 2e-5;
            if !crate::close(g, numeric) {
                return Err(format!("GOAL_NOT_MET: parameter {j}: derivative {g:.9}, central difference {numeric:.9}. Implement both hidden paths in ch07.rs."));
            }
        }
    }
    println!("all nine gradients agree on XOR and the changed soft-target rows");
    // Perturb the asymmetric initialization, not the examples, for a second training run.
    for shift in [0., 0.08] {
        let mut start = Net::new();
        start.hidden_weights[0][0] += shift;
        let model = start.train(&XOR, 10_000, 1., derivative)?;
        let loss = model.loss(&XOR)?;
        println!("initialization shift={shift}: XOR mean BCE={loss:.6}");
        if loss >= 0.02 || XOR.iter().any(|&(x, y)| model.predict(x) != y as u8) {
            return Err("GOAL_NOT_MET: goal: train all four XOR rows with mean BCE < 0.02".into());
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_and_full_gradient() -> Result<(), String> {
        let before = Net::new();
        let after = before.train(&XOR, 100, 1., gradient)?;
        assert!(after.loss(&XOR)? < before.loss(&XOR)?);
        assert!(after.forward([0.2, 0.7]).probability.is_finite());
        assert!(gradient(&before, &[]).is_err());
        assert!(before.step(&XOR, 0., gradient).is_err());
        assert!(before.loss(&[([f64::NAN, 0.], 0.)]).is_err());
        crate::solutions::ch07::check()
    }
}
