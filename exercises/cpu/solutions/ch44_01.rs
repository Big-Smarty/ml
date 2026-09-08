fn exact_top(query: &[f32], docs: &[&[f32]]) -> usize {
    (1..docs.len()).fold(0, |best, i| {
        let score = |d: &[f32]| query.iter().zip(d).map(|(a, b)| a * b).sum::<f32>();
        if score(docs[i]) > score(docs[best]) {
            i
        } else {
            best
        }
    })
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
