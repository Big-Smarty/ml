//! Multifeature linear regression with training-set feature scaling.

const RAW: [([f64; 3], f64); 8] = [
    ([45.0, 1.0, 12.0], 118.0),
    ([60.0, 2.0, 10.0], 157.0),
    ([72.0, 2.0, 8.0], 183.0),
    ([85.0, 3.0, 7.0], 220.0),
    ([95.0, 3.0, 5.0], 238.0),
    ([110.0, 4.0, 4.0], 278.0),
    ([125.0, 4.0, 2.0], 307.0),
    ([140.0, 5.0, 1.0], 347.0),
];

type Example = ([f64; 3], f64);

#[derive(Clone, Copy, Debug)]
struct Scaler {
    mean: [f64; 3],
    scale: [f64; 3],
}

impl Scaler {
    fn fit(rows: &[[f64; 3]]) -> Result<Self, &'static str> {
        if rows.is_empty() {
            return Err("scaling requires training rows");
        }
        if rows.iter().flatten().any(|value| !value.is_finite()) {
            return Err("features must be finite");
        }
        let n = rows.len() as f64;
        let mean = std::array::from_fn(|j| rows.iter().map(|row| row[j]).sum::<f64>() / n);
        let scale = std::array::from_fn(|j| {
            (rows
                .iter()
                .map(|row| (row[j] - mean[j]).powi(2))
                .sum::<f64>()
                / n)
                .sqrt()
        });
        if scale
            .iter()
            .any(|value| *value == 0.0 || !value.is_finite())
        {
            return Err("each feature must vary in the training data");
        }
        Ok(Self { mean, scale })
    }

    fn transform(self, row: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|j| (row[j] - self.mean[j]) / self.scale[j])
    }
}

#[derive(Clone, Copy, Debug)]
struct LinearModel {
    weights: [f64; 3],
    bias: f64,
}

impl LinearModel {
    fn predict(self, features: [f64; 3]) -> f64 {
        self.weights
            .iter()
            .zip(features)
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias
    }

    fn loss_and_gradient(self, data: &[Example]) -> Result<(f64, Self), &'static str> {
        if data.is_empty() {
            return Err("training requires examples");
        }
        let n = data.len() as f64;
        let mut loss = 0.0;
        let mut gradient = Self {
            weights: [0.0; 3],
            bias: 0.0,
        };
        for &(x, y) in data {
            if !y.is_finite() || x.iter().any(|value| !value.is_finite()) {
                return Err("training values must be finite");
            }
            let error = self.predict(x) - y;
            loss += error * error / n;
            for (dw, value) in gradient.weights.iter_mut().zip(x) {
                *dw += 2.0 * error * value / n;
            }
            gradient.bias += 2.0 * error / n;
        }
        if loss.is_finite()
            && gradient.bias.is_finite()
            && gradient.weights.iter().all(|value| value.is_finite())
        {
            Ok((loss, gradient))
        } else {
            Err("loss or gradient overflowed")
        }
    }

    fn train(mut self, data: &[Example], steps: usize, rate: f64) -> Result<Self, &'static str> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        for _ in 0..steps {
            let (_, gradient) = self.loss_and_gradient(data)?;
            for (weight, dw) in self.weights.iter_mut().zip(gradient.weights) {
                *weight -= rate * dw;
            }
            self.bias -= rate * gradient.bias;
            if !self.bias.is_finite() || self.weights.iter().any(|value| !value.is_finite()) {
                return Err("update diverged");
            }
        }
        Ok(self)
    }
}

fn scaled_data() -> Result<(Scaler, Vec<Example>), &'static str> {
    let rows: Vec<_> = RAW.iter().map(|(x, _)| *x).collect();
    let scaler = Scaler::fit(&rows)?;
    let data = RAW.iter().map(|&(x, y)| (scaler.transform(x), y)).collect();
    Ok((scaler, data))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (scaler, data) = scaled_data()?;
    let initial = LinearModel {
        weights: [0.0; 3],
        bias: 0.0,
    };
    let model = initial.train(&data, 2_000, 0.05)?;
    println!("training MSE: {:.4}", model.loss_and_gradient(&data)?.0);
    let home = [100.0, 3.0, 6.0];
    println!(
        "features {:?} -> prediction {:.2}",
        home,
        model.predict(scaler.transform(home))
    );
    println!("weights act on standardized features: {:?}", model.weights);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaling_is_fit_from_rows_and_centers_them() -> Result<(), &'static str> {
        let rows = [[1.0, 10.0, -2.0], [3.0, 20.0, 0.0], [5.0, 30.0, 2.0]];
        let scaler = Scaler::fit(&rows)?;
        let transformed: Vec<_> = rows.into_iter().map(|row| scaler.transform(row)).collect();
        for j in 0..3 {
            assert!(transformed.iter().map(|row| row[j]).sum::<f64>().abs() < 1e-12);
        }
        assert!(Scaler::fit(&[[1.0; 3], [1.0; 3]]).is_err());
        Ok(())
    }

    #[test]
    fn several_measurements_improve_the_fit() -> Result<(), &'static str> {
        let (scaler, data) = scaled_data()?;
        let start = LinearModel {
            weights: [0.0; 3],
            bias: 0.0,
        };
        let model = start.train(&data, 2_000, 0.05)?;
        assert!(model.loss_and_gradient(&data)?.0 < 2.1);
        let prediction = model.predict(scaler.transform([100.0, 3.0, 6.0]));
        assert!((240.0..260.0).contains(&prediction));
        Ok(())
    }
    #[test]
    fn every_parameter_gradient_matches_the_loss() {
        let (_, data) = scaled_data().unwrap();
        let model = LinearModel {
            weights: [0.3, -0.7, 1.2],
            bias: 2.0,
        };
        let analytic = model.loss_and_gradient(&data).unwrap().1;
        for index in 0..4 {
            let mut plus = model;
            let mut minus = model;
            if index < 3 {
                plus.weights[index] += 1e-5;
                minus.weights[index] -= 1e-5;
            } else {
                plus.bias += 1e-5;
                minus.bias -= 1e-5;
            }
            let numeric = (plus.loss_and_gradient(&data).unwrap().0
                - minus.loss_and_gradient(&data).unwrap().0)
                / 2e-5;
            let actual = if index < 3 {
                analytic.weights[index]
            } else {
                analytic.bias
            };
            assert!((numeric - actual).abs() < 1e-6 + 1e-4 * numeric.abs());
        }
    }
}
