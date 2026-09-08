// TODO: return the row-major offset.
fn row_major_offset(row: usize, col: usize, cols: usize) -> usize {
    let _ = (row, col, cols);
    todo!("row-major offset")
}
fn main() {
    println!("{}", row_major_offset(1, 2, 4));
}
#[test]
fn layout() {
    assert_eq!(row_major_offset(1, 2, 4), 6);
    assert_eq!(row_major_offset(2, 1, 3), 7);
    assert_eq!(row_major_offset(0, 2, 3), 2);
}
