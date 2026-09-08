fn output_size(input: usize, kernel: usize, padding: usize, stride: usize) -> usize {
    (input + 2 * padding - kernel) / stride + 1
}
fn main() {
    println!("output side: {}", output_size(5, 3, 1, 1));
}
#[test]
fn checks_stride_and_padding() {
    assert_eq!(output_size(7, 3, 1, 2), 4);
}
