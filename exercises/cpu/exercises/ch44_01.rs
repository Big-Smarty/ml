fn exact_top(_query: &[f32], _docs: &[&[f32]]) -> usize {
    // TODO: return the document index with the largest dot product.
    todo!("scan every document")
}
fn main() {
    println!(
        "top: {}",
        exact_top(&[1.0, 0.0], &[&[0.8, 0.6], &[0.6, 0.8]])
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retrieves_largest_inner_product() {
        assert_eq!(exact_top(&[1.0, 0.0], &[&[0.8, 0.6], &[0.6, 0.8]]), 0);
        assert_eq!(exact_top(&[0.0, 1.0], &[&[0.8, 0.6], &[0.6, 0.8]]), 1);
        assert_eq!(exact_top(&[1.0], &[&[-2.0], &[-1.0], &[-3.0]]), 1);
    }
}
