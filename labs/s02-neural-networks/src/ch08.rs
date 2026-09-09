//! Learner: generalize the one-operation pullback to every reachable operation.
use crate::tape::{Op, Tape};
pub type Backward = fn(&mut Tape, usize) -> Result<(), String>;
pub fn backward(tape: &mut Tape, root: usize) -> Result<(), String> {
    tape.prepare(root)?;
    // Working baseline: sensitivities to the root's immediate operands only.
    let node = tape.nodes[root].clone();
    match node.op {
        Op::Leaf => {}
        Op::Add(a, b) => {
            tape.accumulate(a, 1.)?;
            tape.accumulate(b, 1.)?;
        }
        Op::Mul(a, b) => {
            tape.accumulate(a, tape.value(b)?)?;
            tape.accumulate(b, tape.value(a)?)?;
        }
        Op::Tanh(a) => tape.accumulate(a, 1. - node.value * node.value)?,
    }
    Ok(())
}
pub type Neuron = fn([f64; 2], f64, f64) -> Result<(f64, [f64; 2]), String>;
/// Working affine squared-error neuron. Replace with the tanh computation graph.
pub fn neuron_gradient(
    parameters: [f64; 2],
    input: f64,
    target: f64,
) -> Result<(f64, [f64; 2]), String> {
    if parameters
        .iter()
        .chain([input, target].iter())
        .any(|v| !v.is_finite())
    {
        return Err("finite neuron values required".into());
    }
    let residual = parameters[0] * input + parameters[1] - target;
    let loss = residual * residual;
    let gradient = [2. * residual * input, 2. * residual];
    if !loss.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
        return Err("affine neuron overflow".into());
    }
    Ok((loss, gradient))
}
pub fn train_neuron(
    neuron: Neuron,
    input: f64,
    target: f64,
) -> Result<([f64; 2], f64, f64), String> {
    let mut parameters = [0.5, 0.1];
    let initial = neuron(parameters, input, target)?.0;
    for _ in 0..20 {
        let (_, gradient) = neuron(parameters, input, target)?;
        for (p, g) in parameters.iter_mut().zip(gradient) {
            *p -= 0.1 * g;
        }
    }
    Ok((parameters, initial, neuron(parameters, input, target)?.0))
}
pub fn run(args: &[String]) -> Result<(), String> {
    report(args, backward, neuron_gradient)
}
pub fn check() -> Result<(), String> {
    verify(backward, neuron_gradient)
}
pub fn report(args: &[String], backward: Backward, neuron: Neuron) -> Result<(), String> {
    let input = match args {
        [] => 0.4,
        [flag, value] if flag == "--input" => {
            value.parse::<f64>().map_err(|_| "input must be a number")?
        }
        _ => return Err("08 options: --input NUMBER".into()),
    };
    if !input.is_finite() || input.abs() > 5. {
        return Err("neuron input must be finite with magnitude <=5".into());
    }
    let mut tape = Tape::default();
    let x = tape.leaf(3.)?;
    let m = tape.mul(x, x)?;
    let y = tape.add(m, x)?;
    backward(&mut tape, y)?;
    for (id, n) in tape.nodes.iter().enumerate() {
        println!(
            "node {id}: {:?}, value={}, sensitivity={}",
            n.op, n.value, n.grad
        );
    }
    println!("Full reverse goal: dy/dx=7. Baseline reports only immediate sensitivities.");
    let (loss, gradient) = neuron([0.5, 0.1], input, 0.2)?;
    println!("neuron input={input}, target=0.2: initial loss={loss:.9}, gradient={gradient:?}");
    let (parameters, before, after) = train_neuron(neuron, input, 0.2)?;
    println!("20 neuron updates: loss {before:.9}->{after:.9}, parameters={parameters:?}; learner starts affine, goal is tanh graph");
    Ok(())
}
pub fn verify(backward: Backward, neuron: Neuron) -> Result<(), String> {
    let mut shared = Tape::default();
    let x = shared.leaf(3.)?;
    let m = shared.mul(x, x)?;
    let root = shared.add(m, x)?;
    backward(&mut shared, root)?;
    if shared.nodes[x].grad != 7. {
        return Err(format!(
            "GOAL_NOT_MET: x*x+x at x=3 needs derivative 7, got {}; count all three operand paths",
            shared.nodes[x].grad
        ));
    }
    println!("shared x*x+x: value 12 and derivative 7 agree");

    for xvalue in [3., -0.4, 0.7] {
        let mut tape = Tape::default();
        let x = tape.leaf(xvalue)?;
        let m = tape.mul(x, x)?;
        let y = tape.add(m, x)?;
        let z = tape.tanh(y)?;
        // A later unrelated branch must not receive a gradient.
        let unrelated = tape.mul(m, m)?;
        for _ in 0..2 {
            backward(&mut tape, z)?;
            let f = |v: f64| (v * v + v).tanh();
            let numeric = (f(xvalue + 1e-5) - f(xvalue - 1e-5)) / 2e-5;
            let actual = tape.nodes[x].grad;
            println!(
                "x={xvalue}: tanh(x*x+x) derivative={actual:.9}, central difference={numeric:.9}"
            );
            if !crate::close(actual, numeric) || tape.nodes[unrelated].grad != 0. {
                return Err("GOAL_NOT_MET: goal: propagate all reachable paths, accumulate shared operands, and clear each call; see ch08.rs".into());
            }
        }
    }
    // Two separate equal-valued leaves must remain separate variables.
    let mut t = Tape::default();
    let a = t.leaf(2.)?;
    let b = t.leaf(2.)?;
    let ab = t.mul(a, b)?;
    let root = t.add(ab, a)?;
    backward(&mut t, root)?;
    if t.nodes[a].grad != 3. || t.nodes[b].grad != 2. {
        return Err("GOAL_NOT_MET: node identity is not value equality".into());
    }
    println!("reverse sweep checkpoint passed; checking assembled neuron");
    for (parameters, input, target) in [([0.5, 0.1], 0.4, 0.2), ([-0.3, 0.2], -0.7, 0.6)] {
        let (loss, gradient) = neuron(parameters, input, target)?;
        let expected_loss = |p: [f64; 2]| ((p[0] * input + p[1]).tanh() - target).powi(2);
        if !crate::close(loss, expected_loss(parameters)) {
            return Err(format!("GOAL_NOT_MET: neuron loss {loss:.9}, expected tanh squared error {:.9}; assemble the tanh neuron in neuron_gradient",expected_loss(parameters)));
        }
        for (j, &actual) in gradient.iter().enumerate() {
            let (mut plus, mut minus) = (parameters, parameters);
            plus[j] += 1e-5;
            minus[j] -= 1e-5;
            let numeric = (expected_loss(plus) - expected_loss(minus)) / 2e-5;
            println!("neuron input={input}, parameter {j}: gradient={actual:.9}, central difference={numeric:.9}");
            if !crate::close(actual, numeric) {
                return Err("GOAL_NOT_MET: neuron parameter adjoints disagree with changed-input central differences".into());
            }
        }
        let (_, before, after) = train_neuron(neuron, input, target)?;
        if after >= before {
            return Err(
                "GOAL_NOT_MET: twenty neuron updates did not lower the measured objective".into(),
            );
        }
        println!("neuron input={input}: twenty-step loss {before:.9}->{after:.9}");
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_baseline_graph_boundaries_and_reverse() -> Result<(), String> {
        let mut t = Tape::default();
        let a = t.leaf(2.)?;
        let b = t.leaf(4.)?;
        let y = t.mul(a, b)?;
        backward(&mut t, y)?;
        assert_eq!(t.nodes[a].grad, 4.);
        assert_eq!(t.nodes[b].grad, 2.);
        assert!(t.add(a, 999).is_err());
        assert!(t.leaf(f64::NAN).is_err());
        let huge = t.leaf(f64::MAX)?;
        assert!(t.mul(huge, b).is_err());
        let (loss, g) = neuron_gradient([0.5, 0.1], 0.4, 0.2)?;
        assert!(loss.is_finite() && loss >= 0. && g.iter().all(|v| v.is_finite()));
        let (_, before, after) = train_neuron(neuron_gradient, 0.4, 0.2)?;
        assert!(after < before);
        assert!(neuron_gradient([0.5, 0.1], f64::NAN, 0.2).is_err());
        crate::solutions::ch08::check()
    }
}
