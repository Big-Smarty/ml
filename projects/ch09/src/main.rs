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
        if weights.iter().chain(&bias).any(|x| !x.is_finite()) {
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
        for b in 0..inputs.rows {
            for o in 0..self.out_features {
                let mut sum = self.bias[o];
                for i in 0..self.in_features {
                    sum += inputs.at(b, i) * self.w(o, i)
                }
                outputs.data[b * self.out_features + o] = sum;
            }
        }
        Ok(outputs)
    }
    fn backward(
        &self,
        inputs: &Matrix,
        output_gradients: &Matrix,
    ) -> Result<(Matrix, Vec<f64>, Vec<f64>), String> {
        if inputs.cols != self.in_features
            || output_gradients.rows != inputs.rows
            || output_gradients.cols != self.out_features
        {
            return Err(
                "backward shapes must be inputs=[batch,in_features], output_gradients=[batch,out_features]"
                    .into(),
            );
        }
        let mut input_gradients = Matrix::zeros(inputs.rows, self.in_features);
        let mut weight_gradients = vec![0.0; self.weights.len()];
        let mut bias_gradients = vec![0.0; self.out_features];
        for b in 0..inputs.rows {
            for o in 0..self.out_features {
                let gradient = output_gradients.at(b, o);
                bias_gradients[o] += gradient;
                for i in 0..self.in_features {
                    input_gradients.data[b * self.in_features + i] += gradient * self.w(o, i);
                    weight_gradients[o * self.in_features + i] += gradient * inputs.at(b, i);
                }
            }
        }
        Ok((input_gradients, weight_gradients, bias_gradients))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let inputs = Matrix::new(2, 3, vec![1., 2., 3., 4., 5., 6.])?;
    let layer = Dense::new(3, 2, vec![1., 0., -1., 2., 1., 0.], vec![0.5, -0.5])?;
    let outputs = layer.forward(&inputs)?;
    println!("forward [batch=2,out_features=2]: {:?}", outputs.data);
    let output_gradients = Matrix::new(2, 2, vec![1., 2., 3., 4.])?;
    let (input_gradients, weight_gradients, bias_gradients) =
        layer.backward(&inputs, &output_gradients)?;
    println!(
        "input_gradients={:?}\nweight_gradients={:?}\nbias_gradients={:?}",
        input_gradients.data, weight_gradients, bias_gradients
    );
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
        let (inputs, layer) = fixture();
        assert_eq!(
            layer.forward(&inputs).unwrap().data,
            vec![-1.5, 3.5, -1.5, 12.5]
        );
        let output_gradients = Matrix::new(2, 2, vec![1., 2., 3., 4.]).unwrap();
        let (input_gradients, weight_gradients, bias_gradients) =
            layer.backward(&inputs, &output_gradients).unwrap();
        assert_eq!(input_gradients.data, vec![5., 2., -1., 11., 4., -3.]);
        assert_eq!(weight_gradients, vec![13., 17., 21., 18., 24., 30.]);
        assert_eq!(bias_gradients, vec![4., 6.]);
    }
    #[test]
    fn weight_gradient_matches_difference() {
        let (inputs, layer) = fixture();
        let output_gradients = Matrix::new(2, 2, vec![0.2, -0.3, 0.4, 0.1]).unwrap();
        let (_, weight_gradients, _) = layer.backward(&inputs, &output_gradients).unwrap();
        let score = |dense: &Dense| {
            dense
                .forward(&inputs)
                .unwrap()
                .data
                .iter()
                .zip(&output_gradients.data)
                .map(|(a, b)| a * b)
                .sum::<f64>()
        };
        let h = 1e-5;
        let mut plus = layer.clone();
        let mut minus = layer.clone();
        plus.weights[1] += h;
        minus.weights[1] -= h;
        let numerical = (score(&plus) - score(&minus)) / (2. * h);
        assert!((weight_gradients[1] - numerical).abs() < 1e-6 + 1e-4 * numerical.abs());
    }
    #[test]
    fn bad_shapes_are_errors() {
        let (inputs, layer) = fixture();
        assert!(layer
            .forward(&Matrix::new(1, 2, vec![1., 2.]).unwrap())
            .is_err());
        assert!(layer
            .backward(&inputs, &Matrix::new(1, 2, vec![1., 2.]).unwrap())
            .is_err());
    }
}
