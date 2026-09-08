fn clip(g: &mut [f32], limit: f32) {
    let n = g.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > limit {
        for x in g {
            *x *= limit / n;
        }
    }
}
fn main() {
    println!("Clip one global gradient after accumulation.");
}
#[test]
fn clips_norm() {
    let mut g = [3.0, 4.0];
    clip(&mut g, 1.0);
    assert!((g[0] - 0.6).abs() < 1e-6);
}
