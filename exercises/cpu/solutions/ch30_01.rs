fn output_index(row: usize, col: usize, columns: usize) -> usize {
    row * columns + col
}

fn main() {
    println!("{}", output_index(2, 3, 5));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_index_uses_column_count() {
        assert_eq!(output_index(2, 3, 5), 13);
    }
}
