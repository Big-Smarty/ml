fn adapter_parameters(input: usize, output: usize, rank: usize) -> usize {
    input * rank + rank * output
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
