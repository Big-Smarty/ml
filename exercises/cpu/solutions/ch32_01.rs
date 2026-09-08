fn global_memory_values(elements: usize, fused: bool) -> usize {
    if fused { elements * 2 } else { elements * 4 }
}

fn main() {
    println!("{}", global_memory_values(1024, true));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_removes_two_intermediate_transfers() {
        assert_eq!(global_memory_values(100, false), 400);
        assert_eq!(global_memory_values(100, true), 200);
    }
}
