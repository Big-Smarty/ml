fn causal_allowed(query: usize, key: usize) -> bool {
    // TODO: a query may read its own and earlier keys.
    let _ = (query, key);
    todo!()
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
