//! Three-feature data plumbing. Each row is [raw sensor, load, vibration].
pub type Row = ([f64; 3], f64);
pub type Model = [f64; 4]; // three weights, then bias
pub fn predict(m: Model, x: [f64; 3]) -> f64 {
    m[..3].iter().zip(x).map(|(w, x)| w * x).sum::<f64>() + m[3]
}
pub fn validate(data: &[Row]) -> Result<(), String> {
    if data.is_empty()
        || data
            .iter()
            .any(|(x, y)| !y.is_finite() || x.iter().any(|v| !v.is_finite()))
    {
        Err("expected nonempty finite three-feature rows".into())
    } else {
        Ok(())
    }
}
pub fn loss(m: Model, data: &[Row]) -> Result<f64, String> {
    validate(data)?;
    let l = data
        .iter()
        .map(|&(x, y)| (predict(m, x) - y).powi(2))
        .sum::<f64>()
        / data.len() as f64;
    if l.is_finite() {
        Ok(l)
    } else {
        Err("vector MSE overflow".into())
    }
}
pub fn fixture() -> Vec<Row> {
    let mut rows = Vec::new();
    for x in [-1., 1.] {
        for load in [-1., 1.] {
            for vibration in [-1., 1.] {
                rows.push((
                    [x, 1000. * load, vibration],
                    2. * x + 3. * load - vibration + 1.,
                ));
            }
        }
    }
    rows
}
#[derive(Debug, Clone, Copy)]
pub struct Scaler {
    pub mean: [f64; 3],
    pub scale: [f64; 3],
}
impl Scaler {
    pub fn fit(data: &[Row]) -> Result<Self, String> {
        validate(data)?;
        let n = data.len() as f64;
        let mean = std::array::from_fn(|j| data.iter().map(|(x, _)| x[j]).sum::<f64>() / n);
        let scale = std::array::from_fn(|j| {
            (data
                .iter()
                .map(|(x, _)| (x[j] - mean[j]).powi(2))
                .sum::<f64>()
                / n)
                .sqrt()
        });
        if scale.iter().any(|s| !s.is_finite() || *s == 0.) {
            return Err(
                "each training feature must vary; remove constant columns explicitly".into(),
            );
        }
        Ok(Self { mean, scale })
    }
    pub fn transform(self, x: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|j| (x[j] - self.mean[j]) / self.scale[j])
    }
    pub fn rows(self, data: &[Row]) -> Vec<Row> {
        data.iter().map(|&(x, y)| (self.transform(x), y)).collect()
    }
}
