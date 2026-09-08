fn matmul_ijk(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
}
#[cfg(test)]
fn matmul_ikj(_a: &[f32], _b: &[f32], _c: &mut [f32], _m: usize, _k: usize, _n: usize) {
    // TODO: reorder the loops so both B rows and C rows are contiguous in the inner loop.
    todo!("write the i-p-j kernel")
}
fn main() {
    let a = [1., 2., 3., 4., 5., 6.];
    let b = [1., 2., 3., 4., 5., 6.];
    let mut c = [0.; 4];
    matmul_ijk(&a, &b, &mut c, 2, 3, 2);
    println!("prior scalar matmul: {c:?}; next implement the locality-friendly order");
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reordered_matches_reference() {
        let a = [1., 2., 3., 4., 5., 6.];
        let b = [1., 2., 3., 4., 5., 6.];
        let mut c = [0.; 4];
        matmul_ikj(&a, &b, &mut c, 2, 3, 2);
        assert_eq!(c, [22., 28., 49., 64.]);
    }
}
