fn softmax(logits: [f64; 3]) -> [f64; 3] {
    let maximum = logits.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities = logits.map(|logit| (logit - maximum).exp());
    let sum: f64 = probabilities.iter().sum();
    probabilities.iter_mut().for_each(|value| *value /= sum);
    probabilities
}

fn route(logits: [f64; 3]) -> (usize, f64) {
    let probabilities = softmax(logits);
    let expert = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    (expert, probabilities[expert])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (expert, gate) = route([0.2, 1.1, -0.4]);
    println!("expert {expert}, selected full-softmax gate {gate:.4}");
    let mut trainer = ch56::Trainer::new(ch56::Model::new(ch56::Config::tiny(3), 56)?, 5600);
    let data = b"rust routes words. rust learns.";
    println!(
        "contextual LM task CE before: {:.4}",
        trainer.evaluate(data)?
    );
    trainer.train_step(data, 16, 0.08)?;
    println!(
        "after one complete reference step: {:.4}",
        trainer.evaluate(data)?
    );
    println!(
        "Complete the local selected-gate derivative test, then inspect the library backward pass."
    );
    Ok(())
}

#[cfg(test)]
fn selected_gate_derivative(gate: f64) -> f64 {
    let _ = gate;
    // TODO: retain the full-softmax derivative of the selected gate.
    todo!("use p * (1 - p)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_gate_keeps_router_gradient() {
        let (expert, gate) = route([0.2, 1.1, -0.4]);
        // TODO: derivative of selected softmax probability with respect to its own logit.
        let derivative = selected_gate_derivative(gate);
        assert_eq!(expert, 1);
        assert!(derivative > 0.0 && (derivative - gate * (1.0 - gate)).abs() < 1e-12);
    }
}
