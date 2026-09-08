#[derive(Clone, Copy)]
struct Prediction {
    probability: f64,
    label: u8,
}
const DATA: [Prediction; 2] = [
    Prediction {
        probability: 0.2,
        label: 0,
    },
    Prediction {
        probability: 0.8,
        label: 1,
    },
];

fn brier(data: &[Prediction]) -> f64 {
    assert!(!data.is_empty());
    data.iter()
        .map(|prediction| (prediction.probability - prediction.label as f64).powi(2))
        .sum::<f64>()
        / data.len() as f64
}
fn main() {
    println!("{}", brier(&DATA));
}
#[test]
fn score() {
    assert!((brier(&DATA) - 0.04).abs() < 1e-12);
    assert!(
        (brier(&[Prediction {
            probability: 0.2,
            label: 1,
        }]) - 0.64)
            .abs()
            < 1e-12
    );
}
