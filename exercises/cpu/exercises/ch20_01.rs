fn output_size(_input_size: usize, _kernel_size: usize, _padding: usize, _stride: usize) -> usize {
    // TODO: return floor((input_size + 2*padding - kernel_size) / stride) + 1.
    todo!()
}
fn main() {
    println!("output side: {}", output_size(5, 3, 1, 1));
}
#[test]
fn checks_stride_and_padding() {
    assert_eq!(output_size(7, 3, 1, 2), 4);
}
