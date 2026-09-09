//! The normalization axis is all 36 pixels of ONE image, not a mini-batch.
//! Backward differentiates the mean and variance as well as the division.
use crate::residual::{self, Core, PIXELS};
const SIDE: usize = 6;
pub(crate) const CORE: Core = Core {
    normalize,
    backward: normalize_backward,
    translate,
};

pub(crate) fn normalize(input: &[f64; PIXELS]) -> ([f64; PIXELS], f64) {
    let mean = input.iter().sum::<f64>() / PIXELS as f64;
    let variance = input
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / PIXELS as f64;
    let inv_std = 1.0 / (variance + 1e-5).sqrt();
    (input.map(|value| (value - mean) * inv_std), inv_std)
}

pub(crate) fn normalize_backward(
    output_gradient: &[f64; PIXELS],
    normalized_output: &[f64; PIXELS],
    inv_std: f64,
) -> [f64; PIXELS] {
    let gradient_sum: f64 = output_gradient.iter().sum();
    let gradient_times_output_sum: f64 = output_gradient
        .iter()
        .zip(normalized_output)
        .map(|(gradient, output)| gradient * output)
        .sum();
    std::array::from_fn(|j| {
        inv_std
            * (PIXELS as f64 * output_gradient[j]
                - gradient_sum
                - normalized_output[j] * gradient_times_output_sum)
            / PIXELS as f64
    })
}

pub(crate) fn translate(image: &[f64; PIXELS], dr: isize, dc: isize) -> [f64; PIXELS] {
    let mut out = [0.0; PIXELS];
    for r in 0..SIDE {
        for c in 0..SIDE {
            let sr = r as isize - dr;
            let sc = c as isize - dc;
            if (0..SIDE as isize).contains(&sr) && (0..SIDE as isize).contains(&sc) {
                out[r * SIDE + c] = image[sr as usize * SIDE + sc as usize];
            }
        }
    }
    out
}
pub fn run(args: &[String]) -> Result<(), String> {
    residual::run(CORE, args)
}
pub fn check() -> Result<(), String> {
    residual::check(CORE)
}
