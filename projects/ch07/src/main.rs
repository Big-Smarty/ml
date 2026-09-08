//! A 2-2-1 network that learns XOR by manual backpropagation.
const XOR: [([f64; 2], f64); 4] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[derive(Clone, Debug)]
struct Net {
    w1: [[f64; 2]; 2],
    b1: [f64; 2],
    w2: [f64; 2],
    b2: f64,
}

impl Net {
    fn new() -> Self {
        Self {
            w1: [[0.7, -0.4], [-0.2, 0.9]],
            b1: [0.1, -0.3],
            w2: [0.8, -0.6],
            b2: 0.2,
        }
    }

    fn forward(&self, x: [f64; 2]) -> ([f64; 2], f64, f64) {
        let h = [
            sigmoid(self.w1[0][0] * x[0] + self.w1[0][1] * x[1] + self.b1[0]),
            sigmoid(self.w1[1][0] * x[0] + self.w1[1][1] * x[1] + self.b1[1]),
        ];
        let z = self.w2[0] * h[0] + self.w2[1] * h[1] + self.b2;
        (h, z, sigmoid(z))
    }

    fn loss(&self) -> f64 {
        XOR.iter()
            .map(|&(x, y)| {
                let z = self.forward(x).1;
                z.max(0.0) - z * y + (-z.abs()).exp().ln_1p()
            })
            .sum::<f64>()
            / XOR.len() as f64
    }

    fn step(&mut self, rate: f64) -> Result<(), &'static str> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err("learning rate must be finite and positive");
        }
        let mut gw1 = [[0.0; 2]; 2];
        let mut gb1 = [0.0; 2];
        let mut gw2 = [0.0; 2];
        let mut gb2 = 0.0;
        for &(x, y) in &XOR {
            let (h, _, p) = self.forward(x);
            // Sigmoid plus binary cross-entropy simplifies dL/d(output logit) to p-y.
            let dz2 = p - y;
            for (j, ((gw1_row, gw2_value), gb1_value)) in
                gw1.iter_mut().zip(&mut gw2).zip(&mut gb1).enumerate()
            {
                *gw2_value += dz2 * h[j];
                let dz1 = dz2 * self.w2[j] * h[j] * (1.0 - h[j]);
                *gb1_value += dz1;
                for (gradient, input) in gw1_row.iter_mut().zip(x) {
                    *gradient += dz1 * input;
                }
            }
            gb2 += dz2;
        }
        let scale = rate / XOR.len() as f64;
        for (
            (((weights, weight_gradients), bias), bias_gradient),
            (output_weight, output_gradient),
        ) in self
            .w1
            .iter_mut()
            .zip(gw1)
            .zip(&mut self.b1)
            .zip(gb1)
            .zip(self.w2.iter_mut().zip(gw2))
        {
            for (weight, gradient) in weights.iter_mut().zip(weight_gradients) {
                *weight -= scale * gradient;
            }
            *bias -= scale * bias_gradient;
            *output_weight -= scale * output_gradient;
        }
        self.b2 -= scale * gb2;
        if !self.loss().is_finite() {
            return Err("training produced a nonfinite loss");
        }
        Ok(())
    }
}

fn train(steps: usize, rate: f64) -> Result<Net, &'static str> {
    let mut net = Net::new();
    for _ in 0..steps {
        net.step(rate)?;
    }
    Ok(net)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let net = train(10_000, 1.0)?;
    println!("mean binary cross-entropy: {:.6}", net.loss());
    for &(x, y) in &XOR {
        println!("{:?} target {y:.0} probability {:.4}", x, net.forward(x).2);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learns_xor() {
        let n = train(10_000, 1.0).unwrap();
        assert!(n.loss() < 0.02);
        for &(x, y) in &XOR {
            assert_eq!((n.forward(x).2 >= 0.5) as u8, y as u8);
        }
    }
    #[test]
    fn one_weight_gradient_matches_central_difference() {
        let n = Net::new();
        let h = 1e-5;
        let mut plus = n.clone();
        let mut minus = n.clone();
        plus.w2[0] += h;
        minus.w2[0] -= h;
        let numeric = (plus.loss() - minus.loss()) / (2.0 * h);
        let analytic = XOR
            .iter()
            .map(|&(x, y)| {
                let (hidden, _, p) = n.forward(x);
                (p - y) * hidden[0]
            })
            .sum::<f64>()
            / 4.0;
        assert!((numeric - analytic).abs() < 1e-6 + 1e-4 * numeric.abs());
    }
    #[test]
    fn hidden_weight_gradient_matches_central_difference() {
        let n = Net::new();
        let h = 1e-5;
        let mut p = n.clone();
        let mut m = n.clone();
        p.w1[1][0] += h;
        m.w1[1][0] -= h;
        let numeric = (p.loss() - m.loss()) / (2. * h);
        let analytic = XOR
            .iter()
            .map(|&(x, y)| {
                let (hidden, _, prob) = n.forward(x);
                (prob - y) * n.w2[1] * hidden[1] * (1. - hidden[1]) * x[0]
            })
            .sum::<f64>()
            / 4.;
        assert!((numeric - analytic).abs() < 1e-6 + 1e-4 * numeric.abs());
    }
    #[test]
    fn bad_rate_is_rejected() {
        assert!(Net::new().step(0.0).is_err());
    }
    #[test]
    fn actual_update_matches_all_nine_numerical_gradients() {
        fn parameter(net: &mut Net, index: usize) -> &mut f64 {
            match index {
                0..=3 => &mut net.w1[index / 2][index % 2],
                4..=5 => &mut net.b1[index - 4],
                6..=7 => &mut net.w2[index - 6],
                8 => &mut net.b2,
                _ => unreachable!(),
            }
        }
        let mut old = Net::new();
        let mut updated = old.clone();
        updated.step(0.1).unwrap();
        for index in 0..9 {
            let mut plus = old.clone();
            let mut minus = old.clone();
            *parameter(&mut plus, index) += 1e-5;
            *parameter(&mut minus, index) -= 1e-5;
            let numerical = (plus.loss() - minus.loss()) / 2e-5;
            let actual = (*parameter(&mut old, index) - *parameter(&mut updated, index)) / 0.1;
            assert!((actual - numerical).abs() < 1e-6 + 1e-4 * numerical.abs());
        }
    }
}
