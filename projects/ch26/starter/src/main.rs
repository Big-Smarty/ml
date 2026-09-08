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

fn matmul_scalar_row_col_inner(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    validate_matmul_shapes(a, b, c, m, k, n)?;
    c.fill(0.0);
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    Ok(())
}

#[cfg(test)]
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
    let a = [1., 2., 3., 4., 5., 6.];
    let b = [1., 2., 3., 4., 5., 6.];
    let mut c = [0.; 4];
    matmul_scalar_row_col_inner(&a, &b, &mut c, 2, 3, 2).unwrap();
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
        matmul_scalar_row_inner_col(&a, &b, &mut c, 2, 3, 2).unwrap();
        assert_eq!(c, [22., 28., 49., 64.]);
    }
}
