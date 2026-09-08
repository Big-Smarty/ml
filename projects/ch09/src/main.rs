//! A row-major batched dense layer and its exact backward pass.
#[derive(Clone, Debug, PartialEq)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}
impl Matrix {
    fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, String> {
        if rows == 0 || cols == 0 {
            return Err("matrix dimensions must be positive".into());
        }
        if data.len() != rows.checked_mul(cols).ok_or("matrix size overflow")? {
            return Err("data length does not match shape".into());
        }
        if data.iter().any(|x| !x.is_finite()) {
            return Err("matrix values must be finite".into());
        }
        Ok(Self { rows, cols, data })
    }
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    fn at(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }
}

#[derive(Clone, Debug)]
struct Dense {
    input: usize,
    output: usize,
    weights: Vec<f64>,
    bias: Vec<f64>,
}
impl Dense {
    fn new(input: usize, output: usize, weights: Vec<f64>, bias: Vec<f64>) -> Result<Self, String> {
        if input == 0 || output == 0 {
            return Err("feature counts must be positive".into());
        }
        if weights.len() != input.checked_mul(output).ok_or("weight size overflow")?
            || bias.len() != output
        {
            return Err("parameter length does not match [out,in] and [out]".into());
        }
        if weights.iter().chain(&bias).any(|x| !x.is_finite()) {
            return Err("parameters must be finite".into());
        }
        Ok(Self {
            input,
            output,
            weights,
            bias,
        })
    }
    fn w(&self, o: usize, i: usize) -> f64 {
        self.weights[o * self.input + i]
    }
    fn forward(&self, x: &Matrix) -> Result<Matrix, String> {
        if x.cols != self.input {
            return Err(format!(
                "input has {} features; layer expects {}",
                x.cols, self.input
            ));
        }
        let mut y = Matrix::zeros(x.rows, self.output);
        for b in 0..x.rows {
            for o in 0..self.output {
                let mut sum = self.bias[o];
                for i in 0..self.input {
                    sum += x.at(b, i) * self.w(o, i)
                }
                y.data[b * self.output + o] = sum;
            }
        }
        Ok(y)
    }
    fn backward(&self, x: &Matrix, dy: &Matrix) -> Result<(Matrix, Vec<f64>, Vec<f64>), String> {
        if x.cols != self.input || dy.rows != x.rows || dy.cols != self.output {
            return Err("backward shapes must be x=[batch,in], dy=[batch,out]".into());
        }
        let mut dx = Matrix::zeros(x.rows, self.input);
        let mut dw = vec![0.0; self.weights.len()];
        let mut db = vec![0.0; self.output];
        for b in 0..x.rows {
            for o in 0..self.output {
                let g = dy.at(b, o);
                db[o] += g;
                for i in 0..self.input {
                    dx.data[b * self.input + i] += g * self.w(o, i);
                    dw[o * self.input + i] += g * x.at(b, i);
                }
            }
        }
        Ok((dx, dw, db))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Matrix::new(2, 3, vec![1., 2., 3., 4., 5., 6.])?;
    let layer = Dense::new(3, 2, vec![1., 0., -1., 2., 1., 0.], vec![0.5, -0.5])?;
    let y = layer.forward(&x)?;
    println!("forward [batch=2,out=2]: {:?}", y.data);
    let dy = Matrix::new(2, 2, vec![1., 2., 3., 4.])?;
    let (dx, dw, db) = layer.backward(&x, &dy)?;
    println!("dx={:?}\ndw={:?}\ndb={:?}", dx.data, dw, db);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Matrix, Dense) {
        (
            Matrix::new(2, 3, vec![1., 2., 3., 4., 5., 6.]).unwrap(),
            Dense::new(3, 2, vec![1., 0., -1., 2., 1., 0.], vec![0.5, -0.5]).unwrap(),
        )
    }
    #[test]
    fn hand_computed_forward_and_backward() {
        let (x, l) = fixture();
        assert_eq!(l.forward(&x).unwrap().data, vec![-1.5, 3.5, -1.5, 12.5]);
        let dy = Matrix::new(2, 2, vec![1., 2., 3., 4.]).unwrap();
        let (dx, dw, db) = l.backward(&x, &dy).unwrap();
        assert_eq!(dx.data, vec![5., 2., -1., 11., 4., -3.]);
        assert_eq!(dw, vec![13., 17., 21., 18., 24., 30.]);
        assert_eq!(db, vec![4., 6.]);
    }
    #[test]
    fn weight_gradient_matches_difference() {
        let (x, l) = fixture();
        let dy = Matrix::new(2, 2, vec![0.2, -0.3, 0.4, 0.1]).unwrap();
        let (_, dw, _) = l.backward(&x, &dy).unwrap();
        let score = |d: &Dense| {
            d.forward(&x)
                .unwrap()
                .data
                .iter()
                .zip(&dy.data)
                .map(|(a, b)| a * b)
                .sum::<f64>()
        };
        let h = 1e-5;
        let mut p = l.clone();
        let mut m = l.clone();
        p.weights[1] += h;
        m.weights[1] -= h;
        let n = (score(&p) - score(&m)) / (2. * h);
        assert!((dw[1] - n).abs() < 1e-6 + 1e-4 * n.abs());
    }
    #[test]
    fn bad_shapes_are_errors() {
        let (x, l) = fixture();
        assert!(l
            .forward(&Matrix::new(1, 2, vec![1., 2.]).unwrap())
            .is_err());
        assert!(l
            .backward(&x, &Matrix::new(1, 2, vec![1., 2.]).unwrap())
            .is_err());
    }
}
