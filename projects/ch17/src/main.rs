//! One-component PCA for two-feature data, with explicit covariance and power iteration.

type Vec2 = [f64; 2];
type Mat2 = [[f64; 2]; 2];

fn validate(rows: &[Vec2]) -> Result<(), &'static str> {
    if rows.len() < 2 {
        return Err("PCA requires at least two rows");
    }
    if rows.iter().flatten().any(|x| !x.is_finite()) {
        return Err("PCA inputs must be finite");
    }
    Ok(())
}

fn mean(rows: &[Vec2]) -> Result<Vec2, &'static str> {
    validate(rows)?;
    let sum = rows
        .iter()
        .fold([0.0; 2], |a, x| [a[0] + x[0], a[1] + x[1]]);
    Ok([sum[0] / rows.len() as f64, sum[1] / rows.len() as f64])
}

fn covariance(rows: &[Vec2], mu: Vec2) -> Result<Mat2, &'static str> {
    validate(rows)?;
    let mut c = [[0.0; 2]; 2];
    for x in rows {
        let z = [x[0] - mu[0], x[1] - mu[1]];
        c[0][0] += z[0] * z[0];
        c[0][1] += z[0] * z[1];
        c[1][0] += z[1] * z[0];
        c[1][1] += z[1] * z[1];
    }
    let denominator = (rows.len() - 1) as f64;
    for row in &mut c {
        for value in row {
            *value /= denominator;
        }
    }
    Ok(c)
}

fn mat_vec(a: Mat2, v: Vec2) -> Vec2 {
    [
        a[0][0] * v[0] + a[0][1] * v[1],
        a[1][0] * v[0] + a[1][1] * v[1],
    ]
}

fn dot(a: Vec2, b: Vec2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn power_iteration_from(a: Mat2, steps: usize, mut v: Vec2) -> Result<(Vec2, f64), &'static str> {
    if steps == 0 || a.iter().flatten().any(|x| !x.is_finite()) {
        return Err("power iteration needs finite values and at least one step");
    }
    for _ in 0..steps {
        let next = mat_vec(a, v);
        let norm = next[0].hypot(next[1]);
        if !norm.is_finite() || norm == 0.0 {
            return Err("power iteration reached a zero or non-finite direction");
        }
        v = [next[0] / norm, next[1] / norm];
    }
    let eigenvalue = dot(v, mat_vec(a, v));
    Ok((v, eigenvalue))
}

fn power_iteration(a: Mat2, steps: usize) -> Result<(Vec2, f64), &'static str> {
    let a_result = power_iteration_from(a, steps, [1.0, 0.5]);
    let b_result = power_iteration_from(a, steps, [-0.5, 1.0]);
    match (a_result, b_result) {
        (Ok(a), Ok(b)) => Ok(if a.1 >= b.1 { a } else { b }),
        // One seed may be exactly in the null space of a rank-one covariance.
        (Ok(result), Err(_)) | (Err(_), Ok(result)) => Ok(result),
        (Err(error), Err(_)) => Err(error),
    }
}

#[derive(Clone, Copy, Debug)]
struct Pca1 {
    mean: Vec2,
    component: Vec2,
    eigenvalue: f64,
    covariance: Mat2,
}

impl Pca1 {
    fn fit(rows: &[Vec2]) -> Result<Self, &'static str> {
        let mean = mean(rows)?;
        let covariance = covariance(rows, mean)?;
        // Constant data has no distinguished direction: choose the first axis.
        let (mut component, eigenvalue) = if covariance.iter().flatten().all(|&x| x == 0.0) {
            ([1.0, 0.0], 0.0)
        } else {
            power_iteration(covariance, 40)?
        };
        if component[0] < 0.0 {
            component = [-component[0], -component[1]];
        }
        Ok(Self {
            mean,
            component,
            eigenvalue,
            covariance,
        })
    }

    fn transform(self, x: Vec2) -> f64 {
        dot(self.component, [x[0] - self.mean[0], x[1] - self.mean[1]])
    }

    fn reconstruct(self, score: f64) -> Vec2 {
        [
            self.mean[0] + score * self.component[0],
            self.mean[1] + score * self.component[1],
        ]
    }

    fn reconstruction_mse(self, rows: &[Vec2]) -> f64 {
        rows.iter()
            .map(|&x| {
                let r = self.reconstruct(self.transform(x));
                (x[0] - r[0]).powi(2) + (x[1] - r[1]).powi(2)
            })
            .sum::<f64>()
            / rows.len() as f64
    }

    fn explained_fraction(self) -> f64 {
        let trace = self.covariance[0][0] + self.covariance[1][1];
        // Reporting convention for a data set with no variance to retain.
        if trace == 0.0 {
            0.0
        } else {
            self.eigenvalue / trace
        }
    }

    fn condition_number(self) -> f64 {
        let scale = self
            .covariance
            .iter()
            .flatten()
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        if scale == 0.0 {
            return f64::INFINITY;
        }
        let a = self.covariance[0][0] / scale;
        let b = self.covariance[0][1] / scale;
        let d = self.covariance[1][1] / scale;
        let large = 0.5 * (a + d + (a - d).hypot(2.0 * b));
        // Product of eigenvalues equals the determinant; avoid trace - discriminant cancellation.
        let small = (a * d - b * b) / large;
        if small <= 0.0 {
            f64::INFINITY
        } else {
            large / small
        }
    }
}

const MEASUREMENTS: [Vec2; 6] = [
    [1.0, 1.2],
    [2.0, 1.9],
    [3.0, 3.2],
    [4.0, 3.9],
    [5.0, 5.1],
    [6.0, 5.8],
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pca = Pca1::fit(&MEASUREMENTS)?;
    println!("mean: [{:.3}, {:.3}]", pca.mean[0], pca.mean[1]);
    println!("sample covariance: {:?}", pca.covariance);
    println!(
        "PC1: [{:.4}, {:.4}], eigenvalue {:.4}",
        pca.component[0], pca.component[1], pca.eigenvalue
    );
    println!(
        "variance retained: {:.1}%",
        100.0 * pca.explained_fraction()
    );
    println!("covariance condition number: {:.1}", pca.condition_number());
    let x = MEASUREMENTS[0];
    let score = pca.transform(x);
    println!(
        "first row -> score {score:.4} -> reconstruction {:?}",
        pca.reconstruct(score)
    );
    println!(
        "mean squared reconstruction distance: {:.5}",
        pca.reconstruction_mse(&MEASUREMENTS)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn units_do_not_change_direction_or_conditioning() -> Result<(), &'static str> {
        let original = Pca1::fit(&MEASUREMENTS)?;
        for scale in [1e-10, 1e10] {
            let rows: Vec<_> = MEASUREMENTS
                .iter()
                .map(|x| [scale * x[0], scale * x[1]])
                .collect();
            let scaled = Pca1::fit(&rows)?;
            assert!(dot(original.component, scaled.component).abs() > 1.0 - 1e-10);
            assert!((original.explained_fraction() - scaled.explained_fraction()).abs() < 1e-10);
            assert!((original.condition_number() / scaled.condition_number() - 1.0).abs() < 1e-10);
        }
        let constant = [[3.0, -2.0]; 4];
        let pca = Pca1::fit(&constant)?;
        assert_eq!(pca.eigenvalue, 0.0);
        assert_eq!(pca.explained_fraction(), 0.0);
        assert_eq!(pca.reconstruction_mse(&constant), 0.0);
        Ok(())
    }

    #[test]
    fn covariance_and_power_iteration_match_known_matrix() -> Result<(), &'static str> {
        let a = [[2.0, 1.0], [1.0, 2.0]];
        let (v, lambda) = power_iteration(a, 30)?;
        assert!((lambda - 3.0).abs() < 1e-10);
        assert!((v[0].abs() - 1.0 / 2.0_f64.sqrt()).abs() < 1e-6);
        assert!((dot(v, v) - 1.0).abs() < 1e-12);
        let residual = mat_vec(a, v);
        assert!((residual[0] - lambda * v[0]).hypot(residual[1] - lambda * v[1]) < 1e-8);
        Ok(())
    }

    #[test]
    fn rank_one_data_reconstructs_exactly() -> Result<(), &'static str> {
        for rows in [
            [[-2.0, -4.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 4.0]],
            [[1.0, -2.0], [0.5, -1.0], [-0.5, 1.0], [-1.0, 2.0]],
        ] {
            let pca = Pca1::fit(&rows)?;
            assert!(pca.reconstruction_mse(&rows) < 1e-20);
        }
        Ok(())
    }

    #[test]
    fn projection_is_translation_invariant() -> Result<(), &'static str> {
        let shifted: Vec<_> = MEASUREMENTS
            .iter()
            .map(|x| [x[0] + 10.0, x[1] - 7.0])
            .collect();
        let a = Pca1::fit(&MEASUREMENTS)?;
        let b = Pca1::fit(&shifted)?;
        assert!((a.eigenvalue - b.eigenvalue).abs() < 1e-10);
        assert!(dot(a.component, b.component).abs() > 1.0 - 1e-10);
        assert!(
            (a.reconstruction_mse(&MEASUREMENTS) - b.reconstruction_mse(&shifted)).abs() < 1e-10
        );
        Ok(())
    }

    #[test]
    fn rejects_too_few_or_nonfinite_rows() {
        assert!(Pca1::fit(&[[1.0, 2.0]]).is_err());
        assert!(Pca1::fit(&[[1.0, 2.0], [f64::NAN, 3.0]]).is_err());
    }
}
