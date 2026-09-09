//! Working pointwise digit model: center weight plus top-left subsampling.
//! Replace convolve, pool and backward with shared spatial convolution and max pooling.
//! Parsing, ten-class head, SGD and evaluation are supplied in vision.rs.
use crate::vision::{self, Core};
pub(crate) const CORE: Core = Core {
    convolve,
    pool,
    backward,
};

pub(crate) fn convolve(
    image: &[f64],
    _rows: usize,
    _cols: usize,
    kernel: &[f64; 9],
    bias: f64,
) -> Vec<f64> {
    image.iter().map(|pixel| kernel[4] * pixel + bias).collect()
}
pub(crate) fn pool(values: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<usize>) {
    let mut output = Vec::new();
    let mut winners = Vec::new();
    for r in 0..rows / 2 {
        for c in 0..cols / 2 {
            let at = 2 * r * cols + 2 * c;
            output.push(values[at]);
            winners.push(at);
        }
    }
    (output, winners)
}
pub(crate) fn backward(image: &[f64], _rows: usize, _cols: usize, incoming: &[f64]) -> [f64; 9] {
    let mut gradient = [0.0; 9];
    // Correct derivative of the supplied pointwise baseline.
    gradient[4] = image.iter().zip(incoming).map(|(x, g)| x * g).sum();
    gradient
}
pub fn run(args: &[String]) -> Result<(), String> {
    vision::run(CORE, args)
}
pub fn check() -> Result<(), String> {
    vision::check(CORE)
}
