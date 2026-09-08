#[derive(Clone, Copy)]
struct Point {
    features: [f64; 2],
    label: f64,
}

#[cfg(test)]
fn hinge_loss(score: f64, label: f64) -> f64 {
    let _ = (score, label);
    // TODO: return max(0, 1 - label * score), using the supplied raw score.
    todo!("compute one point's hinge loss")
}

fn main() {
    let p = Point {
        features: [2.0, 1.0],
        label: 1.0,
    };
    let score = 0.25 * p.features[0] + 0.0 * p.features[1];
    println!(
        "known linear score: {score}, signed label {}; now implement hinge_loss for the test",
        p.label
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn active_and_inactive_hinges() {
        let p = Point {
            features: [2.0, 1.0],
            label: 1.0,
        };
        assert_eq!(hinge_loss(0.25 * p.features[0], p.label), 0.5);
        assert_eq!(hinge_loss(p.features[0], p.label), 0.0);
    }
}
