fn dispatch_count(element_count: usize, workgroup_size: usize) -> usize {
    element_count.div_ceil(workgroup_size)
}

fn main() {
    println!("{}", dispatch_count(129, 64));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_exact_and_partial_workgroups() {
        assert_eq!(dispatch_count(128, 64), 2);
        assert_eq!(dispatch_count(129, 64), 3);
    }
}
