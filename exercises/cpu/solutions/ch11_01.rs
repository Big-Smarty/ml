fn momentum(velocity: &mut f64, gradient: f64, beta: f64) -> f64 {
    *velocity = beta * *velocity + gradient;
    *velocity
}
fn main() {
    let mut velocity = 0.0;
    println!("{}", momentum(&mut velocity, 2.0, 0.9));
}
#[test]
fn remembers() {
    let mut velocity = 0.0;
    assert_eq!(momentum(&mut velocity, 2.0, 0.9), 2.0);
    assert_eq!(momentum(&mut velocity, 0.0, 0.9), 1.8);
}
