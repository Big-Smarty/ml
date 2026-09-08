fn squared_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    // TODO: sum the squared coordinate differences. No square root is needed.
    let _ = (a, b);
    todo!("compute squared Euclidean distance")
}

fn main() {
    println!("{}", squared_distance([1.0, 2.0], [4.0, 6.0]));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn distance_matches_three_four_five_triangle() {
        assert_eq!(squared_distance([1.0, 2.0], [4.0, 6.0]), 25.0);
        assert_eq!(squared_distance([2.0, -1.0], [2.0, -1.0]), 0.0);
    }
}
