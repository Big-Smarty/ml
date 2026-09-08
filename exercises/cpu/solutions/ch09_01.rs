fn offset(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}
fn main() {
    println!("{}", offset(1, 2, 4));
}
#[test]
fn layout() {
    assert_eq!(offset(1, 2, 4), 6);
    assert_eq!(offset(2, 1, 3), 7);
    assert_eq!(offset(0, 2, 3), 2);
}
