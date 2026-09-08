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
        if data.iter().any(|value| !value.is_finite()) {
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

    fn at(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }
}

#[derive(Clone, Debug)]
struct Dense {
    in_features: usize,
    out_features: usize,
    weights: Vec<f64>,
    bias: Vec<f64>,
}

impl Dense {
    fn new(
        in_features: usize,
        out_features: usize,
        weights: Vec<f64>,
        bias: Vec<f64>,
    ) -> Result<Self, String> {
        if in_features == 0 || out_features == 0 {
            return Err("feature counts must be positive".into());
        }
        if weights.len()
            != in_features
                .checked_mul(out_features)
                .ok_or("weight size overflow")?
            || bias.len() != out_features
        {
            return Err(
                "parameter length does not match [out_features,in_features] and [out_features]"
                    .into(),
            );
        }
        if weights.iter().chain(&bias).any(|value| !value.is_finite()) {
            return Err("parameters must be finite".into());
        }
        Ok(Self {
            in_features,
            out_features,
            weights,
            bias,
        })
    }

    fn w(&self, out_feature: usize, in_feature: usize) -> f64 {
        self.weights[out_feature * self.in_features + in_feature]
    }

    fn forward(&self, inputs: &Matrix) -> Result<Matrix, String> {
        if inputs.cols != self.in_features {
            return Err(format!(
                "input has {} features; layer expects {}",
                inputs.cols, self.in_features
            ));
        }
        let mut outputs = Matrix::zeros(inputs.rows, self.out_features);
        // TODO: fill outputs with inputs × weights-transpose + bias.
        let _ = &mut outputs;
        todo!("compute inputs times weights-transpose plus bias")
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _guided: fn(&Dense, &Matrix) -> Result<Matrix, String> = Dense::forward;
    let inputs = Matrix::new(2, 3, vec![1., 2., 3., 4., 5., 6.])?;
    let layer = Dense::new(3, 2, vec![1., 0., -1., 2., 1., 0.], vec![0.5, -0.5])?;
    let outputs = Matrix::zeros(inputs.rows, layer.out_features);
    println!(
        "checkpoint: inputs=[{},{}], weights=[{},{}], outputs=[{},{}], first input={}, first weight={}, first bias={}",
        inputs.rows,
        inputs.cols,
        layer.out_features,
        layer.in_features,
        outputs.rows,
        outputs.cols,
        inputs.at(0, 0),
        layer.w(0, 0),
        layer.bias[0]
    );
    Ok(())
}

#[test]
fn dense_forward_matches_worked_example() {
    let inputs = Matrix::new(2, 3, vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let layer = Dense::new(3, 2, vec![1., 0., -1., 2., 1., 0.], vec![0.5, -0.5]).unwrap();
    assert_eq!(
        layer.forward(&inputs).unwrap().data,
        vec![-1.5, 3.5, -1.5, 12.5]
    );
}
