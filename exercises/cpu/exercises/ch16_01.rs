fn hinge_loss(score: f64, label: f64) -> f64 {
    // TODO: the margin is label * score; hinge loss is max(0, 1 - margin).
    let _ = (score, label);
    todo!("compute hinge loss")
}

fn main() {
    println!("{}", hinge_loss(0.25, 1.0));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hinge_is_zero_only_beyond_the_margin() {
        assert_eq!(hinge_loss(2.0, 1.0), 0.0);
        assert_eq!(hinge_loss(0.5, -1.0), 1.5);
    }
}
