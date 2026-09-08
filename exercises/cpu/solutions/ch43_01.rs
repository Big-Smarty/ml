fn adapter_parameters(input_features: usize, output_features: usize, rank: usize) -> usize {
    input_features * rank + rank * output_features
}
fn main() {
    println!("adapter parameters: {}", adapter_parameters(8, 12, 2));
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counts_both_factors() {
        assert_eq!(adapter_parameters(8, 12, 2), 40);
    }
}
