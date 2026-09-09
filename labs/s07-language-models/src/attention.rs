use crate::{ensure, LabResult};
#[derive(Clone, Debug)]
pub struct Inputs {
    pub q: Vec<f64>,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    pub t: usize,
    pub d: usize,
}
#[derive(Debug)]
pub struct Attention {
    pub output: Vec<f64>,
    pub probabilities: Vec<f64>,
}
#[derive(Debug)]
pub struct Gradients {
    pub q: Vec<f64>,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
}
pub type Forward = fn(&Inputs) -> LabResult<Attention>;
pub type Backward = fn(&Inputs, &Attention, &[f64]) -> LabResult<Gradients>;
pub fn validate(x: &Inputs) -> LabResult {
    ensure(
        x.t > 0 && x.t <= 128 && x.d > 0 && x.d <= 128,
        "attention dimensions must be in 1..=128",
    )?;
    ensure(
        [&x.q, &x.k, &x.v]
            .iter()
            .all(|a| a.len() == x.t * x.d && a.iter().all(|v| v.is_finite())),
        "Q K V need finite [T,D] arrays",
    )
}
pub fn validate_backward(x: &Inputs, a: &Attention, up: &[f64]) -> LabResult {
    validate(x)?;
    ensure(
        a.probabilities.len() == x.t * x.t
            && up.len() == x.t * x.d
            && up.iter().chain(&a.probabilities).all(|v| v.is_finite()),
        "backward shape or finite-value mismatch",
    )
}
pub fn fixture() -> Inputs {
    Inputs {
        q: vec![1., 0., 0., 1.],
        k: vec![1., 0., 1., 1.],
        v: vec![2., 0., 0., 4.],
        t: 2,
        d: 2,
    }
}
pub fn run_with(args: &[String], forward: Forward, backward: Backward) -> LabResult {
    crate::no_args(args)?;
    let x = fixture();
    let a = forward(&x)?;
    let g = backward(&x, &a, &[0., 0., 0., 1.])?;
    println!("Q={:?} K={:?} V={:?} [T,D]=[2,2]", x.q, x.k, x.v);
    println!(
        "probabilities={:.6?}; output={:.6?}; dQ={:.6?}",
        a.probabilities, a.output, g.q
    );
    Ok(())
}
pub fn check_with(forward: Forward, backward: Backward) -> LabResult {
    let demo = forward(&fixture())?;
    ensure((demo.output[2]-0.6604769013466862).abs()<1e-9,format!("goal: content-dependent QK scores should give output[2]=0.660476901; got {}. Replace uniform causal averaging.",demo.output[2]))?;
    let x = Inputs {
        q: vec![0.7, -0.2, 0.1, 0.9, -0.5, 0.3],
        k: vec![0.4, 0.6, -0.8, 0.2, 0.3, -0.7],
        v: vec![1., -0.5, 0.2, 0.8, -0.4, 0.9],
        t: 3,
        d: 2,
    };
    let up = [0.3, -0.7, 1.2, 0.4, -0.2, 0.6];
    let a = forward(&x)?;
    let mut changed = x.clone();
    changed.k[4] = 100.;
    changed.v[5] = -100.;
    ensure(
        a.output[..4] == forward(&changed)?.output[..4],
        "goal: future changes affected earlier outputs",
    )?;
    for i in 0..x.t {
        ensure(
            (a.probabilities[i * x.t..(i + 1) * x.t].iter().sum::<f64>() - 1.).abs() < 1e-12,
            "goal: rows sum to one",
        )?;
        for j in i + 1..x.t {
            ensure(
                a.probabilities[i * x.t + j] == 0.,
                "goal: future probability is nonzero",
            )?;
        }
    }
    let g = backward(&x, &a, &up)?;
    for (group, grads) in [&g.q, &g.k, &g.v].iter().enumerate() {
        for (i, &analytic) in grads.iter().enumerate() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            [&mut plus.q, &mut plus.k, &mut plus.v][group][i] += 1e-5;
            [&mut minus.q, &mut minus.k, &mut minus.v][group][i] -= 1e-5;
            let probe = |x: &Inputs| -> LabResult<f64> {
                Ok(forward(x)?.output.iter().zip(up).map(|(a, b)| a * b).sum())
            };
            let numeric = (probe(&plus)? - probe(&minus)?) / 2e-5;
            ensure(
                (analytic - numeric).abs() < 1e-6 + 1e-4 * numeric.abs(),
                format!(
                    "goal: QKV group {group} element {i}: analytic={analytic} numeric={numeric}"
                ),
            )?;
        }
    }
    let twice = backward(&x, &a, &up.map(|v| 2. * v))?;
    for (one, two) in [&g.q, &g.k, &g.v]
        .iter()
        .zip([&twice.q, &twice.k, &twice.v])
    {
        ensure(
            one.iter().zip(two).all(|(a, b)| (2. * a - b).abs() < 1e-12),
            "goal: no extra averaging of upstream gradient",
        )?;
    }
    println!(
        "35 goal passed: scaled QK, causal invariance, every QKV derivative and upstream scaling"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn supplied_baseline_and_full_solution() {
        let x = super::fixture();
        let a = crate::ch35::forward(&x).unwrap();
        assert_eq!(a.output, [2., 0., 1., 2.]);
        super::check_with(
            crate::solutions::ch35::forward,
            crate::solutions::ch35::backward,
        )
        .unwrap();
        let mut bad = x;
        bad.q[0] = f64::NAN;
        assert!(crate::ch35::forward(&bad).is_err());
    }
}
