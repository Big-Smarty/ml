fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn output_delta(_prediction: f32, _target: f32) -> f32 {
    // TODO: differentiate sigmoid cross-entropy.
    todo!("differentiate sigmoid cross-entropy")
}

fn main() {
    let _guided = output_delta;
    let prediction = sigmoid(0.4);
    println!("CPU forward checkpoint: p={prediction:.4}, target=1");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_entropy_sigmoid_delta_has_correct_sign() {
        assert!(
            (output_delta(0.6, 1.0) + 0.4).abs() < 1e-6,
            "guided repair: the output delta must be prediction minus target"
        );
    }
}
