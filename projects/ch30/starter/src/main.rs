use std::{error::Error, io};

fn checked_matmul_lengths(
    m: usize,
    k: usize,
    n: usize,
) -> Result<(usize, usize, usize), Box<dyn Error>> {
    if m == 0 || k == 0 || n == 0 {
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "m, k, and n must be nonzero").into(),
        );
    }
    let overflow = || io::Error::new(io::ErrorKind::InvalidInput, "matrix shape overflows usize");
    Ok((
        m.checked_mul(k).ok_or_else(overflow)?,
        k.checked_mul(n).ok_or_else(overflow)?,
        m.checked_mul(n).ok_or_else(overflow)?,
    ))
}

fn matmul_scalar(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let (a_len, b_len, c_len) = checked_matmul_lengths(m, k, n)?;
    if a.len() != a_len || b.len() != b_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "matrix lengths do not match dimensions",
        )
        .into());
    }
    if a.iter().chain(b).any(|value| !value.is_finite()) {
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "matrix entries must be finite").into(),
        );
    }
    let mut c = vec![0.0; c_len];
    for row in 0..m {
        for col in 0..n {
            c[row * n + col] = (0..k)
                .map(|inner| a[row * k + inner] * b[inner * n + col])
                .sum();
        }
    }
    Ok(c)
}

fn tile_count(_k: usize) -> usize {
    // TODO: include the partial K tile.
    todo!("include the partial K tile")
}

fn main() -> Result<(), Box<dyn Error>> {
    let _guided = tile_count;
    println!(
        "CPU checkpoint: {:?}",
        matmul_scalar(&[1.0, 2.0], &[3.0, 4.0, 5.0, 6.0], 1, 2, 2)?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_inner_dimension_keeps_last_tile() {
        assert_eq!(
            tile_count(17),
            2,
            "guided repair: floor division drops the partial K tile"
        );
    }
}
