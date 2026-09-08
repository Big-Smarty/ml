fn hinge_loss(label: f64, score: f64) -> f64 {
    // TODO: the margin is label * score; hinge loss is max(0, 1 - margin).
    let _ = (label, score);
    todo!("compute hinge loss")
}

fn main() {
    println!("{}", hinge_loss(1.0, 0.25));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hinge_is_zero_only_beyond_the_margin() {
        assert_eq!(hinge_loss(1.0, 2.0), 0.0);
        assert_eq!(hinge_loss(-1.0, 0.5), 1.5);
    }
}
