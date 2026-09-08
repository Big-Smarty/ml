struct Neuron {
    weight: f64,
    bias: f64,
}

impl Neuron {
    fn predict(&self, input: f64) -> f64 {
        self.weight * input + self.bias
    }
}

fn main() {
    let model = Neuron {
        weight: 2.0,
        bias: 1.0,
    };
    println!("{}", model.predict(3.0));
}

#[test]
fn prediction_uses_all_three_values() {
    assert_eq!(
        Neuron {
            weight: 2.0,
            bias: 1.0,
        }
        .predict(3.0),
        7.0
    );
    assert_eq!(
        Neuron {
            weight: -1.0,
            bias: 4.0,
        }
        .predict(2.0),
        2.0
    );
}
