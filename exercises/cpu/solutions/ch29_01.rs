fn dispatch_count(items: usize, workgroup_size: usize) -> usize {
    items.div_ceil(workgroup_size)
}

fn main() {
    println!("{}", dispatch_count(129, 64));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn includes_partial_workgroup() {
        assert_eq!(dispatch_count(128, 64), 2);
        assert_eq!(dispatch_count(129, 64), 3);
    }
}
