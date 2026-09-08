fn offset(row: usize, col: usize, cols: usize) -> usize {
    let _ = (row, col, cols);
    todo!("map a row-major coordinate to its contiguous offset")
}

fn main() {
    let _guided: fn(usize, usize, usize) -> usize = offset;
    let row = [1.0, 2.0, 3.0];
    println!("prior checkpoint: dot = {}", row.iter().sum::<f64>());
}

#[test]
fn row_major_offset() {
    assert_eq!(offset(1, 2, 4), 6);
    assert_eq!(offset(2, 1, 3), 7);
    assert_eq!(offset(0, 2, 3), 2);
}
