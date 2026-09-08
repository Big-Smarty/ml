fn row_ptr(rows: &[&[f64]]) -> Vec<usize> {
    // TODO: append the cumulative nonzero count after each row.
    let _ = rows;
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_rows_keep_the_same_offset() {
        assert_eq!(
            row_ptr(&[&[0.0, 2.0], &[0.0, 0.0], &[3.0, 4.0]]),
            [0, 1, 1, 3]
        );
    }
}
