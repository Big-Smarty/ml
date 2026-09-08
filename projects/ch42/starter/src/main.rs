fn stable_softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<_> = x.iter().map(|v| (v - m).exp()).collect();
    let z = p.iter().sum::<f32>();
    p.iter_mut().for_each(|v| *v /= z);
    p
}
fn online_denominator(_x: &[f32]) -> (f32, f32) {
    // TODO: update running maximum m and shifted denominator l for every score.
    todo!("implement online softmax state")
}
fn main() {
    let _guided: fn(&[f32]) -> (f32, f32) = online_denominator;
    println!(
        "checkpoint stable softmax: {:?}",
        stable_softmax(&[1.0, 3.0, 2.0])
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn online_state_matches_two_pass() {
        let x = [1.0, 3.0, 2.0];
        let (m, l) = online_denominator(&x);
        let p = stable_softmax(&x);
        assert!((m - 3.0).abs() < 1e-6);
        assert!((l - 1.0 / p[1]).abs() < 1e-6);
    }
}
