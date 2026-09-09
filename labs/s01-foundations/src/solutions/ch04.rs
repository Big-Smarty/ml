//! Sigmoid plus BCE differentiates to p-y, with no factor two.
use crate::classifier::{self, Model, Row};
pub fn gradient(m: Model, data: &[Row]) -> Result<Model, String> {
    classifier::loss(m, data)?;
    let mut g = [0.; 3];
    for &(x, y) in data {
        let e = classifier::sigmoid(classifier::logit(m, x)) - f64::from(y);
        for (g, x) in g[..2].iter_mut().zip(x) {
            *g += e * x / data.len() as f64;
        }
        g[2] += e / data.len() as f64;
    }
    if g.iter().any(|v| !v.is_finite()) {
        return Err("classification gradient overflow".into());
    }
    Ok(g)
}
pub fn train(data: &[Row], steps: usize, rate: f64) -> Result<Model, String> {
    crate::scalar::settings(rate)?;
    let mut m = [0.; 3];
    classifier::loss(m, data)?;
    for _ in 0..steps {
        let g = gradient(m, data)?;
        for (p, g) in m.iter_mut().zip(g) {
            *p -= rate * g;
        }
        classifier::loss(m, data)?;
    }
    Ok(m)
}
pub fn run(args: &[String]) -> Result<(), String> {
    crate::ch04::report(train, args)
}
pub fn check() -> Result<(), String> {
    crate::ch04::verify(train)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn check_all_bce_derivatives() -> Result<(), String> {
        let m = [0.3, -0.2, 0.1];
        let g = gradient(m, &classifier::TRAIN)?;
        for j in 0..3 {
            let mut p = m;
            let mut n = m;
            p[j] += 1e-5;
            n[j] -= 1e-5;
            let numeric = (classifier::loss(p, &classifier::TRAIN)?
                - classifier::loss(n, &classifier::TRAIN)?)
                / 2e-5;
            assert!(crate::scalar::close(g[j], numeric));
        }
        Ok(())
    }
}
