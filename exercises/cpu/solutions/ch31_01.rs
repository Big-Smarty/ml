fn output_delta(probability: f32, target: f32) -> f32 {
    probability - target
}

fn main() {
    println!("{}", output_delta(0.6, 1.0));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_moves_wrong_prediction_toward_target() {
        assert!((output_delta(0.6, 1.0) + 0.4).abs() < 1e-6);
        assert!((output_delta(0.6, 0.0) - 0.6).abs() < 1e-6);
    }
}
