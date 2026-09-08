//! Analytical gradients for the scalar line model, checked with central differences.

const TRAIN: [(f64, f64); 5] = [
    (-2.0, -3.0),
    (-1.0, -1.0),
    (0.0, 1.0),
    (1.0, 3.0),
    (2.0, 5.0),
];

#[derive(Clone, Copy, Debug, PartialEq)]
struct Model {
    weight: f64,
    bias: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Gradient {
    weight: f64,
    bias: f64,
}

fn validate(data: &[(f64, f64)]) -> Result<(), &'static str> {
    if data.is_empty() {
        return Err("at least one example is required");
    }
    if data.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
        return Err("examples must be finite");
    }
    Ok(())
}

impl Model {
    fn predict(self, x: f64) -> f64 {
        self.weight * x + self.bias
    }

    fn mse(self, data: &[(f64, f64)]) -> Result<f64, &'static str> {
        validate(data)?;
        let value = data
            .iter()
            .map(|&(x, y)| (self.predict(x) - y).powi(2))
            .sum::<f64>()
            / data.len() as f64;
        value.is_finite().then_some(value).ok_or("loss overflowed")
    }

    fn gradient(self, data: &[(f64, f64)]) -> Result<Gradient, &'static str> {
        validate(data)?;
        let n = data.len() as f64;
        let (weight, bias) = data.iter().fold((0.0, 0.0), |(dw, db), &(x, y)| {
            let error = self.predict(x) - y;
            (dw + 2.0 * error * x / n, db + 2.0 * error / n)
        });
        if weight.is_finite() && bias.is_finite() {
            Ok(Gradient { weight, bias })
        } else {
            Err("gradient overflowed")
        }
    }

    fn numerical_gradient(self, data: &[(f64, f64)]) -> Result<Gradient, &'static str> {
        const H: f64 = 1e-5;
        let weight = (Self {
            weight: self.weight + H,
            ..self
        }
        .mse(data)?
            - Self {
                weight: self.weight - H,
                ..self
            }
            .mse(data)?)
            / (2.0 * H);
        let bias = (Self {
            bias: self.bias + H,
            ..self
        }
        .mse(data)?
            - Self {
                bias: self.bias - H,
                ..self
            }
            .mse(data)?)
            / (2.0 * H);
        Ok(Gradient { weight, bias })
    }

    fn step(self, data: &[(f64, f64)], rate: f64) -> Result<Self, &'static str> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        let gradient = self.gradient(data)?;
        let next = Self {
            weight: self.weight - rate * gradient.weight,
            bias: self.bias - rate * gradient.bias,
        };
        next.mse(data)?;
        Ok(next)
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 + 1e-4 * a.abs().max(b.abs())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Model {
        weight: 0.0,
        bias: 0.0,
    };
    let analytic = model.gradient(&TRAIN)?;
    let numerical = model.numerical_gradient(&TRAIN)?;
    println!(
        "analytic  dw={:.6}, db={:.6}",
        analytic.weight, analytic.bias
    );
    println!(
        "numerical dw={:.6}, db={:.6}",
        numerical.weight, numerical.bias
    );
    println!(
        "check: {}",
        close(analytic.weight, numerical.weight) && close(analytic.bias, numerical.bias)
    );
    for _ in 0..100 {
        model = model.step(&TRAIN, 0.1)?;
    }
    println!(
        "trained weight={:.6}, bias={:.6}, mse={:.10}",
        model.weight,
        model.bias,
        model.mse(&TRAIN)?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytical_gradient_matches_central_difference() -> Result<(), &'static str> {
        for model in [
            Model {
                weight: 0.0,
                bias: 0.0,
            },
            Model {
                weight: 1.3,
                bias: -0.4,
            },
        ] {
            let a = model.gradient(&TRAIN)?;
            let n = model.numerical_gradient(&TRAIN)?;
            assert!(close(a.weight, n.weight));
            assert!(close(a.bias, n.bias));
        }
        Ok(())
    }

    #[test]
    fn one_step_uses_the_expected_derivative() -> Result<(), &'static str> {
        let start = Model {
            weight: 0.0,
            bias: 0.0,
        };
        let gradient = start.gradient(&TRAIN)?;
        assert!(close(gradient.weight, -8.0));
        assert!(close(gradient.bias, -2.0));
        let next = start.step(&TRAIN, 0.1)?;
        assert!(close(next.weight, 0.8));
        assert!(close(next.bias, 0.2));
        assert!(start.mse(&[]).is_err());
        Ok(())
    }
}
