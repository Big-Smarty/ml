fn pooled_accuracy(folds: &[(usize, usize)]) -> f64 {
    let correct: usize = folds.iter().map(|x| x.0).sum();
    let total: usize = folds.iter().map(|x| x.1).sum();
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
