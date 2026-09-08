fn sigmoid(logit: f32) -> f32 {
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exp = logit.exp();
        exp / (1.0 + exp)
    }
}

fn output_delta(_probability: f32, _target: f32) -> f32 {
    // TODO: differentiate sigmoid cross-entropy.
    todo!("differentiate sigmoid cross-entropy")
}

fn main() {
    let _guided = output_delta;
    let probability = sigmoid(0.4);
    println!("CPU forward checkpoint: p={probability:.4}, target=1");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_entropy_sigmoid_delta_has_correct_sign() {
        assert!(
            (output_delta(0.6, 1.0) + 0.4).abs() < 1e-6,
            "guided repair: the output delta must be probability minus target"
        );
    }
}
