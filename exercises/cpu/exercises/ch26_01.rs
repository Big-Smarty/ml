fn matmul_ikj(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    c.fill(0.0);
    // TODO: use i-p-j loops so the innermost loop walks contiguous B and C rows.
    let _ = (a, b, m, k, n);
    todo!("implement the reordered kernel")
}

fn main() {
    let (a, b, mut c) = ([1., 2., 3., 4., 5., 6.], [1., 2., 3., 4., 5., 6.], [0.; 4]);
    matmul_ikj(&a, &b, &mut c, 2, 3, 2);
    println!("{c:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplies_a_rectangular_pair() {
        let (a, b, mut c) = ([1., 2., 3., 4., 5., 6.], [1., 2., 3., 4., 5., 6.], [0.; 4]);
        matmul_ikj(&a, &b, &mut c, 2, 3, 2);
        assert_eq!(c, [22., 28., 49., 64.]);
    }
}
