//! Learner: turn a bias-only batch operation into dense forward AND backward kernels.
use crate::tensor::{Dense, Gradients, Matrix};
pub type Forward = fn(&Dense, &Matrix) -> Result<Matrix, String>;
pub type Backward = fn(&Dense, &Matrix, &Matrix) -> Result<Gradients, String>;
pub fn forward(layer: &Dense, x: &Matrix) -> Result<Matrix, String> {
    layer.validate(x, None)?;
    // A real constant predictor: the same bias row for every example.
    Matrix::new(x.rows, layer.bias.len(), layer.bias.repeat(x.rows))
}
pub fn backward(layer: &Dense, x: &Matrix, dy: &Matrix) -> Result<Gradients, String> {
    layer.validate(x, Some(dy))?;
    let mut bias = vec![0.; layer.bias.len()];
    for row in dy.data.chunks(dy.cols) {
        for (sum, &g) in bias.iter_mut().zip(row) {
            *sum += g;
        }
    }
    Ok(Gradients {
        inputs: Matrix::zeros(x.rows, x.cols)?,
        weights: Matrix::zeros(layer.weights.rows, layer.weights.cols)?,
        bias,
    })
}
pub fn run(args: &[String]) -> Result<(), String> {
    report(args, forward, backward)
}
pub fn check() -> Result<(), String> {
    verify(forward, backward)
}
fn fixture() -> Result<(Matrix, Dense, Matrix), String> {
    Ok((
        Matrix::new(2, 3, vec![1., 2., 3., 4., 5., 6.])?,
        Dense::new(
            Matrix::new(2, 3, vec![1., 0., -1., 2., 1., 0.])?,
            vec![0.5, -0.5],
        )?,
        Matrix::new(2, 2, vec![1., 2., 3., 4.])?,
    ))
}
pub fn report(args: &[String], forward: Forward, backward: Backward) -> Result<(), String> {
    let scale = match args {
        [] => 1.,
        [flag, value] if flag == "--scale" => {
            value.parse::<f64>().map_err(|_| "scale must be a number")?
        }
        _ => return Err("09 options: --scale NUMBER".into()),
    };
    if !scale.is_finite() || scale.abs() > 5. {
        return Err("scale must be finite with magnitude <=5".into());
    }
    let (mut x, w, dy) = fixture()?;
    for v in &mut x.data {
        *v *= scale;
    }
    println!("input multiplier={scale}");
    let y = forward(&w, &x)?;
    let g = backward(&w, &x, &dy)?;
    println!(
        "X [2,3]={:?}; W [2,3]={:?}; bias={:?}",
        x.data, w.weights.data, w.bias
    );
    println!(
        "Y [2,2]={:?}; dX={:?}; dW={:?}; db={:?}",
        y.data, g.inputs.data, g.weights.data, g.bias
    );
    let doubled = Matrix::new(x.rows * 2, x.cols, x.data.repeat(2))?;
    let doubled_dy = Matrix::new(dy.rows * 2, dy.cols, dy.data.repeat(2))?;
    let doubled_g = backward(&w, &doubled, &doubled_dy)?;
    let half_dy = Matrix::new(dy.rows, dy.cols, dy.data.iter().map(|v| v * 0.5).collect())?;
    let half_g = backward(&w, &x, &half_dy)?;
    println!(
        "duplicate batch: dW={:?}, db={:?}; half dY: dW={:?}, db={:?}",
        doubled_g.weights.data, doubled_g.bias, half_g.weights.data, half_g.bias
    );
    Ok(())
}
pub fn verify(forward: Forward, backward: Backward) -> Result<(), String> {
    let (x, layer, dy) = fixture()?;
    let y = forward(&layer, &x)?;
    let g = backward(&layer, &x, &dy)?;
    if y.data != [-1.5, 3.5, -1.5, 12.5]
        || g.inputs.data != [5., 2., -1., 11., 4., -3.]
        || g.weights.data != [13., 17., 21., 18., 24., 30.]
        || g.bias != [4., 6.]
    {
        return Err(format!("GOAL_NOT_MET: dense goal: Y [-1.5,3.5,-1.5,12.5], dX [5,2,-1,11,4,-3], dW [13,17,21,18,24,30], db [4,6]; got Y {:?}, gradients {g:?}",y.data));
    }
    let x = Matrix::new(3, 2, vec![0.3, -0.8, 1.2, 0.4, -0.5, 0.9])?;
    let layer = Dense::new(
        Matrix::new(4, 2, vec![0.2, -0.1, 0.4, 0.7, -0.3, 0.6, 0.8, 0.1])?,
        vec![0.1, -0.2, 0.3, 0.4],
    )?;
    let dy = Matrix::new(
        3,
        4,
        vec![
            0.1, -0.4, 0.2, 0.3, 0.6, 0.2, -0.1, 0.5, -0.3, 0.7, 0.1, -0.2,
        ],
    )?;
    let g = backward(&layer, &x, &dy)?;
    let score = |w: &Dense, x: &Matrix| -> Result<f64, String> {
        Ok(forward(w, x)?
            .data
            .iter()
            .zip(&dy.data)
            .map(|(a, b)| a * b)
            .sum())
    };
    for j in 0..layer.weights.data.len() + layer.bias.len() + x.data.len() {
        let (mut wp, mut wm) = (layer.clone(), layer.clone());
        let (mut xp, mut xm) = (x.clone(), x.clone());
        let nw = layer.weights.data.len();
        let nb = layer.bias.len();
        let actual = if j < nw {
            wp.weights.data[j] += 1e-5;
            wm.weights.data[j] -= 1e-5;
            g.weights.data[j]
        } else if j < nw + nb {
            wp.bias[j - nw] += 1e-5;
            wm.bias[j - nw] -= 1e-5;
            g.bias[j - nw]
        } else {
            xp.data[j - nw - nb] += 1e-5;
            xm.data[j - nw - nb] -= 1e-5;
            g.inputs.data[j - nw - nb]
        };
        let numeric = (score(&wp, &xp)? - score(&wm, &xm)?) / 2e-5;
        if !crate::close(actual, numeric) {
            return Err(format!(
                "GOAL_NOT_MET: rectangular transfer derivative {j}: {actual} versus {numeric}"
            ));
        }
    }
    println!("dense forward/backward and all 18 rectangular transfer derivatives agree");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bias_baseline_shapes_and_dense() -> Result<(), String> {
        let (x, w, dy) = fixture()?;
        let y = forward(&w, &x)?;
        assert_eq!((y.rows, y.cols), (2, 2));
        assert!(y.data.iter().all(|v| v.is_finite()));
        assert_eq!(backward(&w, &x, &dy)?.bias, [4., 6.]);
        assert!(Matrix::new(1, 3, vec![0.; 2]).is_err());
        assert!(Matrix::new(1, 1, vec![f64::NAN]).is_err());
        assert!(forward(&w, &Matrix::zeros(1, 2)?).is_err());
        assert!(backward(&w, &x, &Matrix::zeros(1, 2)?).is_err());
        crate::solutions::ch09::check()
    }
}
