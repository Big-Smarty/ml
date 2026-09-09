//! A dense operation reuses each weight across rows; backward sums those uses.
use crate::tensor::{Dense, Gradients, Matrix};
pub fn forward(layer: &Dense, x: &Matrix) -> Result<Matrix, String> {
    layer.validate(x, None)?;
    let mut y = Matrix::zeros(x.rows, layer.weights.rows)?;
    for row in 0..x.rows {
        for (out, &bias) in layer.bias.iter().enumerate() {
            let mut value = bias;
            for input in 0..x.cols {
                value += x.at(row, input) * layer.weights.at(out, input);
            }
            y.data[row * layer.weights.rows + out] = value;
        }
    }
    Matrix::new(y.rows, y.cols, y.data)
}
pub fn backward(layer: &Dense, x: &Matrix, dy: &Matrix) -> Result<Gradients, String> {
    layer.validate(x, Some(dy))?;
    let mut dx = Matrix::zeros(x.rows, x.cols)?;
    let mut dw = Matrix::zeros(layer.weights.rows, layer.weights.cols)?;
    let mut db = vec![0.; layer.bias.len()];
    for row in 0..x.rows {
        for (out, bias_gradient) in db.iter_mut().enumerate() {
            let g = dy.at(row, out);
            *bias_gradient += g;
            for input in 0..x.cols {
                dx.data[row * x.cols + input] += g * layer.weights.at(out, input);
                dw.data[out * x.cols + input] += g * x.at(row, input);
            }
        }
    }
    if dx
        .data
        .iter()
        .chain(&dw.data)
        .chain(&db)
        .any(|v| !v.is_finite())
    {
        return Err("dense backward overflow".into());
    }
    Ok(Gradients {
        inputs: dx,
        weights: dw,
        bias: db,
    })
}
pub fn run(args: &[String]) -> Result<(), String> {
    crate::ch09::report(args, forward, backward)
}
pub fn check() -> Result<(), String> {
    crate::ch09::verify(forward, backward)
}
