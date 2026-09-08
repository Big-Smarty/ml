#[derive(Clone, Copy)]
struct Point {
    x: [f64; 2],
    y: f64,
}

#[cfg(test)]
fn hinge_loss(weight: [f64; 2], bias: f64, point: Point) -> f64 {
    let _ = (weight, bias, point);
    // TODO: return max(0, 1 - y * (w dot x + b)).
    todo!("compute one point's hinge loss")
}

fn main() {
    let p = Point {
        x: [2.0, 1.0],
        y: 1.0,
    };
    let score = 0.25 * p.x[0] + 0.0 * p.x[1];
    println!(
        "known linear score: {score}, target {}; now implement hinge_loss for the test",
        p.y
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn active_and_inactive_hinges() {
        let p = Point {
            x: [2.0, 1.0],
            y: 1.0,
        };
        assert_eq!(hinge_loss([0.25, 0.0], 0.0, p), 0.5);
        assert_eq!(hinge_loss([1.0, 0.0], 0.0, p), 0.0);
    }
}
