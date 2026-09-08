fn storage_value_access_count(element_count: usize, fused: bool) -> usize {
    if fused {
        element_count * 2
    } else {
        element_count * 4
    }
}

fn main() {
    println!("{}", storage_value_access_count(1024, true));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_removes_intermediate_storage_accesses() {
        assert_eq!(storage_value_access_count(100, false), 400);
        assert_eq!(storage_value_access_count(100, true), 200);
    }
}
