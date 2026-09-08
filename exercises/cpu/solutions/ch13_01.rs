fn fill_missing(value: Option<f64>, training_median: f64) -> f64 {
    value.unwrap_or(training_median)
}
fn main() {
    println!("{}", fill_missing(None, 3.0));
}
#[test]
fn fills_only_missing() {
    assert_eq!(fill_missing(None, 3.0), 3.0);
    assert_eq!(fill_missing(Some(9.0), 3.0), 9.0);
}
