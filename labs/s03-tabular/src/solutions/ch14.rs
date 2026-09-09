//! kNN retains observations; NB compresses them into different class distributions.
use crate::ch14::GaussianNb;
use crate::data::{self, Point, Result};
pub fn knn(rows: &[Point], query: &[f64], k: usize) -> Result<f64> {
    let d = data::validate_points(rows)?;
    data::query(query, d)?;
    if k == 0 || k > rows.len() {
        return Err("k must be in 1..=training count".into());
    }
    let mut distances = rows
        .iter()
        .enumerate()
        .map(|(i, r)| Ok((data::distance(&r.features, query)?, i, r.label)))
        .collect::<Result<Vec<_>>>()?;
    // ponytail: full sorting is clear for 144 rows; partial selection if the table grows.
    distances.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    Ok(distances
        .iter()
        .take(k)
        .map(|r| f64::from(r.2))
        .sum::<f64>()
        / k as f64)
}
pub fn bayes(rows: &[Point], floor: f64) -> Result<GaussianNb> {
    let mut model = crate::ch14::initial_bayes(rows, floor)?;
    let mut count = [0; 2];
    for r in rows {
        count[r.label as usize] += 1;
    }
    for (c, &class_count) in count.iter().enumerate() {
        model.prior[c] = class_count as f64 / rows.len() as f64;
        for (j, v) in model.variance[c].iter_mut().enumerate() {
            *v = data::finite(
                rows.iter()
                    .filter(|r| r.label as usize == c)
                    .map(|r| (r.features[j] - model.mean[c][j]).powi(2))
                    .sum::<f64>()
                    / class_count as f64,
            )?
            .max(floor);
        }
    }
    Ok(model)
}
pub fn run(_: &[String]) -> Result<()> {
    crate::ch14::report(knn, bayes)
}
pub fn check() -> Result<()> {
    crate::ch14::verify(knn, bayes)
}
