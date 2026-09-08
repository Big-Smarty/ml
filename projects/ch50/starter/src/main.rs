#[derive(Clone, Copy, Debug)]
struct Example {
    features: [f64; 4],
    target: f64,
}

const DATA: [Example; 4] = [
    Example {
        features: [-1.0, 0.0, 1.0, 0.5],
        target: -2.0,
    },
    Example {
        features: [0.0, 1.0, -1.0, 1.0],
        target: 2.5,
    },
    Example {
        features: [1.0, 0.5, 0.0, -1.0],
        target: 1.0,
    },
    Example {
        features: [2.0, -1.0, 0.5, 0.0],
        target: 5.0,
    },
];

#[derive(Clone, Copy, Debug)]
struct Model {
    weights: [f64; 4],
    bias: f64,
}

impl Model {
    fn predict(&self, features: &[f64; 4]) -> f64 {
        self.weights
            .iter()
            .zip(features)
            .map(|(weight, feature)| weight * feature)
            .sum::<f64>()
            + self.bias
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Gradient {
    weights: [f64; 4],
    bias: f64,
    count: usize,
}

fn gradient_sum(model: &Model, batch: &[Example]) -> Gradient {
    let mut gradient = Gradient {
        weights: [0.0; 4],
        bias: 0.0,
        count: batch.len(),
    };
    for example in batch {
        let residual = model.predict(&example.features) - example.target;
        for (weight_sum, feature) in gradient.weights.iter_mut().zip(example.features) {
            *weight_sum += 2.0 * residual * feature;
        }
        gradient.bias += 2.0 * residual;
    }
    gradient
}

#[cfg(test)]
fn aggregate_gradient_sums(parts: &[Gradient]) -> Result<Gradient, &'static str> {
    // TODO: sum every weight and bias gradient, and preserve the total example count.
    let _ = parts;
    todo!("aggregate gradient sums and example counts")
}

fn main() {
    let model = Model {
        weights: [0.0; 4],
        bias: 0.0,
    };
    let left = gradient_sum(&model, &DATA[..1]);
    let right = gradient_sum(&model, &DATA[1..]);
    println!(
        "worker 0: weight sums={:?}, bias sum={:.1}, examples={}",
        left.weights, left.bias, left.count
    );
    println!(
        "worker 1: weight sums={:?}, bias sum={:.1}, examples={}",
        right.weights, right.bias, right.count
    );
    println!("Complete the aggregation test, then divide its sums by the returned count.");
}

#[test]
fn worker_aggregation_matches_full_batch() -> Result<(), &'static str> {
    let model = Model {
        weights: [0.0; 4],
        bias: 0.0,
    };
    let parts = [
        gradient_sum(&model, &DATA[..1]),
        gradient_sum(&model, &DATA[1..]),
    ];
    assert_eq!(
        aggregate_gradient_sums(&parts)?,
        gradient_sum(&model, &DATA)
    );
    assert!(aggregate_gradient_sums(&[]).is_err());
    Ok(())
}
