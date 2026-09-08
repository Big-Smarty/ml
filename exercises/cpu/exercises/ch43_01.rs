fn adapter_parameters(_input_features: usize, _output_features: usize, _rank: usize) -> usize {
    // TODO: count A[input_features, rank] plus B[rank, output_features].
    todo!("count LoRA parameters")
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
