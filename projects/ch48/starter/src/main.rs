//! Inspect stable router probabilities before implementing the selected gate.

fn probabilities(logits: [f64; 3]) -> [f64; 3] {
    let max = logits.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let exponentials = logits.map(|z| (z - max).exp());
    let total = exponentials.iter().sum::<f64>();
    exponentials.map(|x| x / total)
}

fn selected_gate_output(probability: f64, expert_output: f64) -> f64 {
    // TODO: multiply the selected probability by its expert output.
    let _ = (probability, expert_output);
    todo!("retain the selected full-softmax probability")
}

fn main() {
    let _guided_todo: fn(f64, f64) -> f64 = selected_gate_output;
    let p = probabilities([1.0, 0.0, -1.0]);
    println!("router probabilities: {p:.3?}; selected expert 0 output: 3");
    println!("Now run cargo test and implement selected_gate_output.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_gate_preserves_probability() {
        assert!((selected_gate_output(0.25, 8.0) - 2.0).abs() < 1e-12);
    }
}
