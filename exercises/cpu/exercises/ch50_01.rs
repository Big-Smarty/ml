fn average(parts: &[(f64, usize)]) -> f64 {
    // TODO: divide summed worker gradients by the summed example counts.
    let _ = parts;
    todo!("divide the total sum by the total count")
}
fn main() {
    println!("Aggregate sufficient statistics, not worker means.");
}
#[test]
fn uneven_workers() {
    assert_eq!(average(&[(4.0, 1), (6.0, 3)]), 2.5);
}
