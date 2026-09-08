fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}
fn exact_top(_query: &[f32], _docs: &[&[f32]]) -> usize {
    // TODO: call dot for every document and return the index with the largest score.
    todo!("scan every document")
}
fn main() {
    let _guided: fn(&[f32], &[f32]) -> f32 = dot;
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
        assert_eq!(dot(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
        assert_eq!(exact_top(&[1.0, 0.0], &[&[0.8, 0.6], &[0.6, 0.8]]), 0);
        assert_eq!(exact_top(&[0.0, 1.0], &[&[0.8, 0.6], &[0.6, 0.8]]), 1);
        assert_eq!(exact_top(&[1.0], &[&[-2.0], &[-1.0], &[-3.0]]), 1);
    }
}
