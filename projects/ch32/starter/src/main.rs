fn storage_value_access_count(_element_count: usize, _fused: bool) -> usize {
    // TODO: count each storage-buffer value read and write.
    todo!("count storage-buffer value accesses")
}

fn affine_relu_scalar(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|value| (value * 1.5 + 0.25).max(0.0))
        .collect()
}

fn main() {
    let _guided = storage_value_access_count;
    let input = [-1.0_f32, 0.0, 2.0];
    println!("CPU scalar oracle: {:?}", affine_relu_scalar(&input));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_removes_intermediate_storage_accesses() {
        assert_eq!(
            storage_value_access_count(100, false),
            400,
            "guided repair: separate kernels write and read the intermediate"
        );
        assert_eq!(
            storage_value_access_count(100, true),
            200,
            "guided repair: fused kernels read input and write output once"
        );
    }
}
