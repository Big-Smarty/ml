fn row_ptr(rows: &[&[f64]]) -> Vec<usize> {
    let mut result = vec![0];
    for row in rows {
        result
            .push(result.last().copied().unwrap_or(0) + row.iter().filter(|&&x| x != 0.0).count());
    }
    result
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
