//! Linear hinge-loss training and an RBF-kernel perceptron, using only std.

#[derive(Clone, Copy, Debug)]
struct Point {
    x: [f64; 2],
    y: f64,
}

fn validate(data: &[Point]) -> Result<(), &'static str> {
    if data.is_empty() {
        return Err("training requires at least one point");
    }
    if data
        .iter()
        .any(|p| !p.x[0].is_finite() || !p.x[1].is_finite() || !p.y.is_finite() || p.y.abs() != 1.0)
    {
        return Err("features must be finite and labels must be -1 or +1");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct LinearSvm {
    w: [f64; 2],
    b: f64,
}

impl LinearSvm {
    fn score(self, x: [f64; 2]) -> f64 {
        self.w[0] * x[0] + self.w[1] * x[1] + self.b
    }

    fn predict(self, x: [f64; 2]) -> f64 {
        if self.score(x) >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    fn objective(self, data: &[Point], lambda: f64) -> Result<f64, &'static str> {
        validate(data)?;
        if !lambda.is_finite() || lambda < 0.0 {
            return Err("lambda must be finite and non-negative");
        }
        if self.w.iter().any(|v| !v.is_finite()) || !self.b.is_finite() {
            return Err("model parameters must be finite");
        }
        let hinge = data
            .iter()
            .map(|p| (1.0 - p.y * self.score(p.x)).max(0.0))
            .sum::<f64>()
            / data.len() as f64;
        let loss = 0.5 * lambda * (self.w[0] * self.w[0] + self.w[1] * self.w[1]) + hinge;
        if !loss.is_finite() {
            return Err("objective overflow; rescale data or reduce the rate");
        }
        Ok(loss)
    }

    fn fit(data: &[Point], epochs: usize, rate: f64, lambda: f64) -> Result<Self, &'static str> {
        validate(data)?;
        if epochs == 0 || !rate.is_finite() || rate <= 0.0 || !lambda.is_finite() || lambda < 0.0 {
            return Err("epochs and finite non-negative training settings are required");
        }
        let mut model = Self {
            w: [0.0; 2],
            b: 0.0,
        };
        for _ in 0..epochs {
            for p in data {
                let margin = p.y * model.score(p.x);
                model.w[0] *= 1.0 - rate * lambda;
                model.w[1] *= 1.0 - rate * lambda;
                if margin < 1.0 {
                    model.w[0] += rate * p.y * p.x[0];
                    model.w[1] += rate * p.y * p.x[1];
                    model.b += rate * p.y;
                }
            }
        }
        model.objective(data, lambda)?;
        Ok(model)
    }
}

fn squared_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

fn rbf(a: [f64; 2], b: [f64; 2], gamma: f64) -> f64 {
    (-gamma * squared_distance(a, b)).exp()
}

struct KernelClassifier<'a> {
    data: &'a [Point],
    alpha: Vec<f64>,
    gamma: f64,
}

impl<'a> KernelClassifier<'a> {
    fn fit(data: &'a [Point], epochs: usize, gamma: f64) -> Result<Self, &'static str> {
        validate(data)?;
        if epochs == 0 || !gamma.is_finite() || gamma <= 0.0 {
            return Err("epochs and a finite positive gamma are required");
        }
        let mut model = Self {
            data,
            alpha: vec![0.0; data.len()],
            gamma,
        };
        for _ in 0..epochs {
            for (i, point) in data.iter().enumerate() {
                if point.y * model.score(point.x) <= 0.0 {
                    model.alpha[i] += 1.0;
                }
            }
        }
        Ok(model)
    }

    fn score(&self, x: [f64; 2]) -> f64 {
        self.data
            .iter()
            .zip(&self.alpha)
            .map(|(p, &a)| a * p.y * rbf(p.x, x, self.gamma))
            .sum()
    }

    fn predict(&self, x: [f64; 2]) -> f64 {
        if self.score(x) >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    fn support_count(&self) -> usize {
        self.alpha.iter().filter(|&&a| a > 0.0).count()
    }
}

fn accuracy(data: &[Point], predict: impl Fn([f64; 2]) -> f64) -> f64 {
    data.iter().filter(|p| predict(p.x) == p.y).count() as f64 / data.len() as f64
}

const LINEAR: [Point; 8] = [
    Point {
        x: [-2.0, -1.0],
        y: -1.0,
    },
    Point {
        x: [-1.5, 0.0],
        y: -1.0,
    },
    Point {
        x: [-1.0, -0.5],
        y: -1.0,
    },
    Point {
        x: [-0.5, -1.5],
        y: -1.0,
    },
    Point {
        x: [0.5, 1.5],
        y: 1.0,
    },
    Point {
        x: [1.0, 0.5],
        y: 1.0,
    },
    Point {
        x: [1.5, 0.0],
        y: 1.0,
    },
    Point {
        x: [2.0, 1.0],
        y: 1.0,
    },
];

const XOR: [Point; 8] = [
    Point {
        x: [-1.2, -0.8],
        y: 1.0,
    },
    Point {
        x: [-0.8, -1.2],
        y: 1.0,
    },
    Point {
        x: [0.8, 1.2],
        y: 1.0,
    },
    Point {
        x: [1.2, 0.8],
        y: 1.0,
    },
    Point {
        x: [-1.2, 0.8],
        y: -1.0,
    },
    Point {
        x: [-0.8, 1.2],
        y: -1.0,
    },
    Point {
        x: [0.8, -1.2],
        y: -1.0,
    },
    Point {
        x: [1.2, -0.8],
        y: -1.0,
    },
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let linear = LinearSvm::fit(&LINEAR, 80, 0.05, 0.01)?;
    println!("linear SVM: w={:?}, b={:.3}", linear.w, linear.b);
    println!(
        "linear fixture accuracy: {:.1}%",
        100.0 * accuracy(&LINEAR, |x| linear.predict(x))
    );
    println!(
        "hinge-plus-penalty objective: {:.4}",
        linear.objective(&LINEAR, 0.01)?
    );

    let straight_on_xor = LinearSvm::fit(&XOR, 80, 0.05, 0.01)?;
    let kernel = KernelClassifier::fit(&XOR, 12, 1.0)?;
    println!(
        "linear accuracy on curved/XOR fixture: {:.1}%",
        100.0 * accuracy(&XOR, |x| straight_on_xor.predict(x))
    );
    println!(
        "RBF kernel accuracy: {:.1}% ({} support points)",
        100.0 * accuracy(&XOR, |x| kernel.predict(x)),
        kernel.support_count()
    );
    println!("Synthetic fixtures verify mechanics; they do not estimate real-world accuracy.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_divergent_finite_settings() {
        let data = [Point {
            x: [2.0, 1.0],
            y: 1.0,
        }];
        assert!(LinearSvm::fit(&data, 2, f64::MAX, 1.0).is_err());
    }

    #[test]
    fn hinge_and_linear_fit_match_hand_checks() -> Result<(), &'static str> {
        let zero = LinearSvm {
            w: [0.0; 2],
            b: 0.0,
        };
        assert!((zero.objective(&LINEAR, 0.01)? - 1.0).abs() < 1e-12);
        let model = LinearSvm::fit(&LINEAR, 80, 0.05, 0.01)?;
        assert_eq!(accuracy(&LINEAR, |x| model.predict(x)), 1.0);
        Ok(())
    }

    #[test]
    fn kernel_handles_xor_where_a_line_does_not() -> Result<(), &'static str> {
        let linear = LinearSvm::fit(&XOR, 80, 0.05, 0.01)?;
        assert!(accuracy(&XOR, |x| linear.predict(x)) <= 0.75);
        let kernel = KernelClassifier::fit(&XOR, 12, 1.0)?;
        assert_eq!(accuracy(&XOR, |x| kernel.predict(x)), 1.0);
        assert!(kernel.support_count() > 0);
        Ok(())
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert!(LinearSvm::fit(&[], 1, 0.1, 0.0).is_err());
        assert!(KernelClassifier::fit(&LINEAR, 1, 0.0).is_err());
    }
}
