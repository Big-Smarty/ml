fn training_median(values: &[Option<f64>]) -> f64 {
    // TODO: fit the median from present training values.
    let _ = values;
    todo!("collect present training values, sort them, and return the middle value")
}
fn main() {
    let _guided_step = training_median;
    let values = [Some(2.0), None, Some(6.0), Some(4.0)];
    println!(
        "missing rows: {}",
        values.iter().filter(|v| v.is_none()).count()
    );
}
#[test]
fn ignores_missing() {
    assert_eq!(
        training_median(&[Some(2.0), None, Some(6.0), Some(4.0)]),
        4.0
    );
    assert_eq!(
        training_median(&[Some(9.0), Some(1.0), None, Some(3.0)]),
        3.0
    );
}
