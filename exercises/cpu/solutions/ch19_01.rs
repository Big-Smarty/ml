fn pooled_accuracy(fold_counts: &[(usize, usize)]) -> f64 {
    let correct: usize = fold_counts.iter().map(|&(correct, _)| correct).sum();
    let total: usize = fold_counts.iter().map(|&(_, total)| total).sum();
    correct as f64 / total as f64
}

fn main() {
    println!("{}", pooled_accuracy(&[(18, 24), (21, 24)]));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_validation_row_has_equal_weight() {
        assert!((pooled_accuracy(&[(8, 10), (1, 2)]) - 0.75).abs() < 1e-12);
        assert!((pooled_accuracy(&[(18, 24), (21, 24)]) - 0.8125).abs() < 1e-12);
    }
}
