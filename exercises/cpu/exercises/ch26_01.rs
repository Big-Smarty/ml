fn validate_matmul_shapes(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    if m == 0
        || k == 0
        || n == 0
        || a.len() != m.checked_mul(k).ok_or("shape overflow")?
        || b.len() != k.checked_mul(n).ok_or("shape overflow")?
        || c.len() != m.checked_mul(n).ok_or("shape overflow")?
    {
        Err("expected A=[m,k], B=[k,n], C=[m,n] with positive dimensions".into())
    } else {
        Ok(())
    }
}

fn matmul_scalar_row_inner_col(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    validate_matmul_shapes(a, b, c, m, k, n)?;
    c.fill(0.0);
    // TODO: use row-inner-col loops so the innermost loop walks contiguous B and C rows.
    todo!("implement the reordered kernel")
}

fn main() {
    let (a, b, mut c) = ([1., 2., 3., 4., 5., 6.], [1., 2., 3., 4., 5., 6.], [0.; 4]);
    matmul_scalar_row_inner_col(&a, &b, &mut c, 2, 3, 2).unwrap();
    println!("{c:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplies_a_rectangular_pair() {
        let (a, b, mut c) = ([1., 2., 3., 4., 5., 6.], [1., 2., 3., 4., 5., 6.], [0.; 4]);
        matmul_scalar_row_inner_col(&a, &b, &mut c, 2, 3, 2).unwrap();
        assert_eq!(c, [22., 28., 49., 64.]);
    }
}
