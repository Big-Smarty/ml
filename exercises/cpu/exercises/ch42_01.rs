fn update(_m: f32, _l: f32, _score: f32) -> (f32, f32) {
    // TODO: update online-softmax maximum and shifted denominator.
    todo!("online update")
}
fn main() {
    println!("state: {:?}", update(1.0, 1.0, 3.0));
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rescales_old_terms_when_max_changes() {
        let (m, l) = update(1.0, 1.0, 3.0);
        assert!((m - 3.0).abs() < 1e-6);
        assert!((l - (1.0 + (-2.0f32).exp())).abs() < 1e-6);
    }
}
