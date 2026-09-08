#[derive(Clone, Copy, Debug)]
struct Neuron {
    weight: f64,
    bias: f64,
}

impl Neuron {
    fn predict(&self, input: f64) -> f64 {
        self.weight * input + self.bias
    }

    fn loss(&self, data: &[(f64, f64)]) -> Result<f64, &'static str> {
        if data.is_empty() {
            return Err("loss requires at least one example");
        }
        if !self.weight.is_finite()
            || !self.bias.is_finite()
            || data
                .iter()
                .any(|(input, target)| !input.is_finite() || !target.is_finite())
        {
            return Err("model and examples must contain finite numbers");
        }
        let loss = data
            .iter()
            .map(|&(input, target)| (self.predict(input) - target).powi(2))
            .sum::<f64>()
            / data.len() as f64;
        if !loss.is_finite() {
            return Err("loss overflowed; reduce input scale or learning rate");
        }
        Ok(loss)
    }
}

fn regularized_weight_gradient(data_gradient: f64, weight: f64, l2: f64) -> f64 {
    let _ = (data_gradient, weight, l2);
    todo!("add the derivative of l2 * weight^2")
}

fn main() -> Result<(), &'static str> {
    let model = Neuron {
        weight: 0.0,
        bias: 0.0,
    };
    let data = [(0.0, 1.0), (1.0, 3.0)];
    let _guided_todo: fn(f64, f64, f64) -> f64 = regularized_weight_gradient;
    println!("prior checkpoint: data loss is {:.3}", model.loss(&data)?);
    println!("worked data gradient for the next weight update: -3");
    println!("Run cargo test to implement the new L2-gradient TODO.");
    Ok(())
}

#[test]
fn l2_adds_two_lambda_weight() {
    assert!((regularized_weight_gradient(-3.0, 2.0, 0.1) + 2.6).abs() < 1e-12);
    assert_eq!(regularized_weight_gradient(3.0, -2.0, 0.0), 3.0);
    assert!((regularized_weight_gradient(0.0, -2.0, 0.1) + 0.4).abs() < 1e-12);
}
