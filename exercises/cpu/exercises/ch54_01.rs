fn rate(successes: usize, eligible: usize) -> Option<f64> {
    // TODO: return no rate for an empty denominator.
    let _ = (successes, eligible);
    todo!("guard the denominator")
}
fn main() {
    println!("Every fairness rate needs its denominator named.");
}
#[test]
fn defined_and_undefined_rates() {
    assert_eq!(rate(3, 4), Some(0.75));
    assert_eq!(rate(0, 0), None);
}
