fn selected_gate(probability: f64, expert_output: f64) -> f64 {
    // TODO: retain the full-softmax probability in the top-1 output.
    let _ = (probability, expert_output);
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_is_not_renormalized_to_one() {
        assert!((selected_gate(0.25, 8.0) - 2.0).abs() < 1e-12);
    }
}
