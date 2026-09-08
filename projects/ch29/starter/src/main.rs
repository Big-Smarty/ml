use std::io;

fn validate_add_inputs(left: &[f32], right: &[f32]) -> io::Result<()> {
    if left.is_empty() || left.len() != right.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "vectors must have the same nonzero length",
        ));
    }
    if left.iter().chain(right).any(|value| !value.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "inputs must be finite",
        ));
    }
    Ok(())
}

fn add_scalar(left: &[f32], right: &[f32]) -> io::Result<Vec<f32>> {
    validate_add_inputs(left, right)?;
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect())
}

fn dispatch_count(element_count: usize, workgroup_size: usize) -> usize {
    // TODO: round up dispatch count.
    let _ = (element_count, workgroup_size);
    todo!("round up dispatch count")
}

fn main() -> io::Result<()> {
    let _guided = dispatch_count;
    let left = [1.0_f32, -2.0, 3.5];
    let right = [2.0_f32, 5.0, -0.5];
    let output = add_scalar(&left, &right)?;
    println!("CPU scalar oracle before GPU dispatch: {output:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_group_is_not_dropped() {
        assert_eq!(
            dispatch_count(67, 64),
            2,
            "guided repair: floor division drops the partial workgroup"
        );
    }

    #[test]
    fn scalar_oracle_maps_left_and_right_to_output() -> io::Result<()> {
        assert_eq!(add_scalar(&[1.0, -2.0], &[2.0, 5.0])?, [3.0, 3.0]);
        Ok(())
    }
}
