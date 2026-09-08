fn cosine(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1];
    let an = (a[0] * a[0] + a[1] * a[1]).sqrt();
    let bn = (b[0] * b[0] + b[1] * b[1]).sqrt();
    dot / (an * bn)
}
fn main() {
    println!("Retrieval compares embeddings in one shared space.");
}
#[test]
fn orthogonal_and_equal() {
    assert_eq!(cosine([1.0, 0.0], [0.0, 2.0]), 0.0);
    assert!((cosine([2.0, 0.0], [3.0, 0.0]) - 1.0).abs() < 1e-12);
}
