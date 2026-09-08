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
    // TODO: return the mean squared probability error.
    let squared_errors = data
        .iter()
        .map(|prediction| (prediction.probability - prediction.label as f64).powi(2));
    let _ = squared_errors;
    todo!("average the squared errors over prediction rows")
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
