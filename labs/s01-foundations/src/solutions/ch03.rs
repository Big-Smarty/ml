//! Every feature receives the SAME example error, times its own feature value.
use crate::vector::{self, Model, Row};
pub fn gradient(m: Model, data: &[Row]) -> Result<Model, String> {
    vector::loss(m, data)?;
    let mut g = [0.; 4];
    for &(x, y) in data {
        let e = vector::predict(m, x) - y;
        for (g, x) in g[..3].iter_mut().zip(x) {
            *g += 2. * e * x / data.len() as f64;
        }
        g[3] += 2. * e / data.len() as f64;
    }
    if g.iter().any(|v| !v.is_finite()) {
        return Err("gradient overflow".into());
    }
    Ok(g)
}
pub fn run(args: &[String]) -> Result<(), String> {
    crate::ch03::report(gradient, args)
}
pub fn check() -> Result<(), String> {
    crate::ch03::verify(gradient)
}
