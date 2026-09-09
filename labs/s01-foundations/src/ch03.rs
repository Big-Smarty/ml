//! Learner: replace the temperature-only gradient with the full vector gradient.
use crate::vector::{self, Model, Row, Scaler};
pub fn gradient(model: Model, data: &[Row]) -> Result<Model, String> {
    vector::loss(model, data)?;
    let mut g = [0.; 4];
    for &(x, y) in data {
        let e = vector::predict(model, x) - y;
        g[0] += 2. * e * x[0] / data.len() as f64;
        g[3] += 2. * e / data.len() as f64;
    }
    Ok(g)
}
pub type GradientFn = fn(Model, &[Row]) -> Result<Model, String>;
pub fn train(data: &[Row], steps: usize, rate: f64, gradient: GradientFn) -> Result<Model, String> {
    crate::scalar::settings(rate)?;
    let mut m = [0.; 4];
    vector::loss(m, data)?;
    for _ in 0..steps {
        let g = gradient(m, data)?;
        for (p, g) in m.iter_mut().zip(g) {
            *p -= rate * g;
        }
        vector::loss(m, data)?;
    }
    Ok(m)
}
pub fn report(g: GradientFn, args: &[String]) -> Result<(), String> {
    let o = crate::args::Options::parse(args, &["--rate", "--steps", "--load-scale"])?;
    let load_scale = o.number("--load-scale", 1000.)?;
    if load_scale <= 0. {
        return Err("load scale must be positive".into());
    }
    let mut raw = vector::fixture();
    for (x, _) in &mut raw {
        x[1] *= load_scale / 1000.;
    }
    let scaler = Scaler::fit(&raw)?;
    let data = scaler.rows(&raw);
    let steps = o.count("--steps", 150)?;
    let rate = o.number("--rate", 0.1)?;
    let model = train(&data, steps, rate, g)?;
    println!("feature order=[raw sensor,load,vibration]; scaler={scaler:?}");
    println!(
        "weights+bias={model:?}, MSE={:.6}, unseen [0.5,half the load scale,-0.5] predicts {:.6}",
        vector::loss(model, &data)?,
        vector::predict(model, scaler.transform([0.5, load_scale * 0.5, -0.5]))
    );
    match train(&raw, steps, rate, g) {
        Ok(m) => println!("raw-units comparison MSE={:.6}", vector::loss(m, &raw)?),
        Err(e) => println!("raw-units comparison: {e}"),
    };
    Ok(())
}
pub fn verify(g: GradientFn) -> Result<(), String> {
    let raw = vector::fixture();
    let s = Scaler::fit(&raw)?;
    let data = s.rows(&raw);
    let m = [0.3, -0.2, 0.7, 0.4];
    let actual = g(m, &data)?;
    for j in 0..4 {
        let mut p = m;
        let mut n = m;
        p[j] += 1e-5;
        n[j] -= 1e-5;
        let numeric = (vector::loss(p, &data)? - vector::loss(n, &data)?) / 2e-5;
        if !crate::scalar::close(actual[j], numeric) {
            return Err(format!(
                "GOAL_NOT_MET: component {j} is {}, finite difference is {numeric}",
                actual[j]
            ));
        }
    }
    let learned = train(&data, 150, 0.1, g)?;
    let error = (vector::predict(learned, s.transform([0.5, 500., -0.5])) - 4.).abs();
    println!("all four gradient components agree; unfamiliar prediction absolute error={error:.9}");
    if error > 1e-5 {
        return Err("GOAL_NOT_MET: held-out dot-product failed".into());
    }
    Ok(())
}
pub fn run(args: &[String]) -> Result<(), String> {
    report(gradient, args)
}
pub fn check() -> Result<(), String> {
    verify(gradient)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_scaling_and_solution() -> Result<(), String> {
        let raw = vector::fixture();
        let s = Scaler::fit(&raw)?;
        let data = s.rows(&raw);
        assert!(vector::loss(train(&data, 150, 0.1, gradient)?, &data)? <= 10.00001);
        assert_eq!(vector::predict([2., 3., -1., 1.], [0.5, 0.5, -0.5]), 4.);
        assert!(Scaler::fit(&[([1.; 3], 0.)]).is_err());
        assert!(gradient([0.; 4], &[]).is_err());
        for j in 0..3 {
            assert!(data.iter().map(|(x, _)| x[j]).sum::<f64>().abs() < 1e-12);
        }
        crate::solutions::ch03::check()
    }
}
