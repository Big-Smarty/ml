fn cosine(a: [f64; 2], b: [f64; 2]) -> f64 {
    // TODO: return dot(a,b) divided by both vector norms.
    let _ = (a, b);
    todo!("normalize the dot product")
}
fn main() {
    println!("Retrieval compares embeddings in one shared space.");
}
#[test]
fn orthogonal_and_equal() {
    assert_eq!(cosine([1.0, 0.0], [0.0, 2.0]), 0.0);
    assert!((cosine([2.0, 0.0], [3.0, 0.0]) - 1.0).abs() < 1e-12);
}
