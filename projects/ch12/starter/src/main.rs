#[derive(Clone, Copy)]
struct Prediction {
    probability: f64,
    label: u8,
}
fn brier(data: &[Prediction]) -> f64 {
    // TODO: implement the probability-sensitive score.
    let _ = data;
    todo!("average the squared difference between probability and label")
}
fn main() {
    let _guided_step = brier;
    let data = [
        Prediction {
            probability: 0.8,
            label: 1,
        },
        Prediction {
            probability: 0.3,
            label: 0,
        },
    ];
    let correct = data
        .iter()
        .filter(|p| (p.probability >= 0.5) == (p.label == 1))
        .count();
    println!(
        "threshold accuracy: {:.3}",
        correct as f64 / data.len() as f64
    );
}
#[test]
fn hand_checked_score() {
    let d = [
        Prediction {
            probability: 0.2,
            label: 0,
        },
        Prediction {
            probability: 0.8,
            label: 1,
        },
    ];
    assert!((brier(&d) - 0.04).abs() < 1e-12);
    assert!(
        (brier(&[Prediction {
            probability: 0.2,
            label: 1
        }]) - 0.64)
            .abs()
            < 1e-12
    );
}
