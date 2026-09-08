fn causal_allowed(query_position: usize, key_position: usize) -> bool {
    key_position <= query_position
}
fn main() {
    println!("Causal masking protects the next-token target.");
}
#[test]
fn lower_triangle() {
    assert!(causal_allowed(2, 0));
    assert!(causal_allowed(2, 2));
    assert!(!causal_allowed(2, 3));
}
