fn squared_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
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
