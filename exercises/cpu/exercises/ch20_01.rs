fn output_size(_input: usize, _kernel: usize, _padding: usize, _stride: usize) -> usize {
    // TODO: return floor((input + 2*padding - kernel) / stride) + 1.
    todo!()
}
fn main() {
    println!("output side: {}", output_size(5, 3, 1, 1));
}
#[test]
fn checks_stride_and_padding() {
    assert_eq!(output_size(7, 3, 1, 2), 4);
}
