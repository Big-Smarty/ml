fn adapter_parameters(_input: usize, _output: usize, _rank: usize) -> usize {
    // TODO: count A[input, rank] plus B[rank, output].
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
