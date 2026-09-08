fn dispatch_count(_len: usize, _workgroup_size: usize) -> usize {
    // TODO: round up dispatch count.
    todo!("round up dispatch count")
}

fn main() {
    let _guided = dispatch_count;
    let a = [1.0_f32, -2.0, 3.5];
    let b = [2.0_f32, 5.0, -0.5];
    let cpu: Vec<_> = a.iter().zip(b).map(|(x, y)| x + y).collect();
    println!("CPU checkpoint before GPU dispatch: {cpu:?}");
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
}
