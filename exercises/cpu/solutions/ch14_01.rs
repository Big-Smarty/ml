fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}
fn main() {
    println!("{}", squared_distance(&[1., 2.], &[4., 6.]));
}
#[test]
fn distance() {
    assert_eq!(squared_distance(&[1., 2.], &[4., 6.]), 25.0);
    assert_eq!(squared_distance(&[2., -1.], &[2., -1.]), 0.0);
    assert_eq!(squared_distance(&[2., -1.], &[-1., -1.]), 9.0);
}
