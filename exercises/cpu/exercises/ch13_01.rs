fn fill_missing(value: Option<f64>, training_median: f64) -> f64 {
    // TODO: use the value when present and the fitted median otherwise.
    let _ = (value, training_median);
    todo!("use the observed value or the median learned from training rows")
}
fn main() {
    println!("{}", fill_missing(None, 3.0));
}
#[test]
fn fills_only_missing() {
    assert_eq!(fill_missing(None, 3.0), 3.0);
    assert_eq!(fill_missing(Some(9.0), 3.0), 9.0);
}
