//! Supplied contiguous rank-two storage and shape validation.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}
impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, String> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(data.len()) {
            return Err("positive dimensions must match storage length".into());
        }
        if data.iter().any(|x| !x.is_finite()) {
            return Err("matrix values must be finite".into());
        }
        Ok(Self { rows, cols, data })
    }
    pub fn zeros(rows: usize, cols: usize) -> Result<Self, String> {
        let size = rows.checked_mul(cols).ok_or("matrix dimensions overflow")?;
        Self::new(rows, cols, vec![0.; size])
    }
    pub fn at(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }
}
#[derive(Clone, Debug)]
pub struct Dense {
    pub weights: Matrix,
    pub bias: Vec<f64>,
}
impl Dense {
    pub fn new(weights: Matrix, bias: Vec<f64>) -> Result<Self, String> {
        if weights.rows != bias.len() || bias.iter().any(|x| !x.is_finite()) {
            return Err("bias must be finite and match output width".into());
        }
        Ok(Self { weights, bias })
    }
    pub fn validate(&self, x: &Matrix, dy: Option<&Matrix>) -> Result<(), String> {
        if x.cols != self.weights.cols {
            return Err("input width must match weights [out,in]".into());
        }
        if let Some(dy) = dy {
            if dy.rows != x.rows || dy.cols != self.weights.rows {
                return Err("upstream gradient must be [batch,out]".into());
            }
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct Gradients {
    pub inputs: Matrix,
    pub weights: Matrix,
    pub bias: Vec<f64>,
}
