fn transferred_floats(_len: usize, _fused: bool) -> usize {
    // TODO: count intermediate reads and writes.
    todo!("count intermediate reads and writes")
}

fn main() {
    let _guided = transferred_floats;
    let input = [-1.0_f32, 0.0, 2.0];
    let output: Vec<_> = input.iter().map(|x| (x * 1.5 + 0.25).max(0.0)).collect();
    println!("CPU checkpoint: {output:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_removes_intermediate_round_trip() {
        assert_eq!(
            transferred_floats(100, false),
            400,
            "guided repair: separate kernels transfer the intermediate twice"
        );
        assert_eq!(
            transferred_floats(100, true),
            200,
            "guided repair: fused kernels read input and write output once"
        );
    }
}
