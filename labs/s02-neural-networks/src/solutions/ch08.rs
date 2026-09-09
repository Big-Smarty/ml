//! Every consumer runs before its parents. The incoming adjoint scales local rules.
use crate::tape::{Op, Tape};
pub fn backward(tape: &mut Tape, root: usize) -> Result<(), String> {
    let order = tape.prepare(root)?;
    for id in order {
        let node = tape.nodes[id].clone();
        let g = node.grad;
        match node.op {
            Op::Leaf => {}
            Op::Add(a, b) => {
                tape.accumulate(a, g)?;
                tape.accumulate(b, g)?;
            }
            Op::Mul(a, b) => {
                // Save both forward values; repeated operand IDs still get two additions.
                let (av, bv) = (tape.value(a)?, tape.value(b)?);
                tape.accumulate(a, g * bv)?;
                tape.accumulate(b, g * av)?;
            }
            Op::Tanh(a) => tape.accumulate(a, g * (1. - node.value * node.value))?,
        }
    }
    Ok(())
}
/// Build exactly the executed scalar recipe, sharing the residual node twice.
pub fn neuron_gradient(
    parameters: [f64; 2],
    input: f64,
    target: f64,
) -> Result<(f64, [f64; 2]), String> {
    let mut tape = Tape::default();
    let w = tape.leaf(parameters[0])?;
    let b = tape.leaf(parameters[1])?;
    let x = tape.leaf(input)?;
    let target = tape.leaf(target)?;
    let negative = tape.leaf(-1.)?;
    let wx = tape.mul(w, x)?;
    let pre = tape.add(wx, b)?;
    let hidden = tape.tanh(pre)?;
    let minus_target = tape.mul(negative, target)?;
    let residual = tape.add(hidden, minus_target)?;
    let loss = tape.mul(residual, residual)?;
    backward(&mut tape, loss)?;
    Ok((tape.value(loss)?, [tape.nodes[w].grad, tape.nodes[b].grad]))
}
pub fn run(args: &[String]) -> Result<(), String> {
    crate::ch08::report(args, backward, neuron_gradient)
}
pub fn check() -> Result<(), String> {
    crate::ch08::verify(backward, neuron_gradient)
}
