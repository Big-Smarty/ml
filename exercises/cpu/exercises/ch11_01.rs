// TODO: update and return the momentum velocity.
fn momentum(v: &mut f64, g: f64, beta: f64) -> f64 {
    let _ = (v, g, beta);
    todo!("store and return beta*v + g")
}
fn main() {
    let mut v = 0.0;
    println!("{}", momentum(&mut v, 2.0, 0.9));
}
#[test]
fn remembers() {
    let mut v = 0.0;
    assert_eq!(momentum(&mut v, 2.0, 0.9), 2.0);
    assert_eq!(momentum(&mut v, 0.0, 0.9), 1.8);
}
