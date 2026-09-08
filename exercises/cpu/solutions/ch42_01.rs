fn update(m: f32, l: f32, score: f32) -> (f32, f32) {
    let next = m.max(score);
    (next, l * (m - next).exp() + (score - next).exp())
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
