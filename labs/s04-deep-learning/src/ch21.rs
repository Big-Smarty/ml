//! Runnable residual model with centering and canonical training views.
//! Implement variance scaling, its coupled derivative, and zero-padded translations.
use crate::residual::{self, Core, PIXELS};
pub(crate) const CORE: Core = Core {
    normalize,
    backward,
    translate,
};

pub(crate) fn normalize(input: &[f64; PIXELS]) -> ([f64; PIXELS], f64) {
    let mean = input.iter().sum::<f64>() / PIXELS as f64;
    // The baseline centers but does not standardize; its scale is one.
    (input.map(|value| value - mean), 1.0)
}
pub(crate) fn backward(
    incoming: &[f64; PIXELS],
    _normalized: &[f64; PIXELS],
    _inv_std: f64,
) -> [f64; PIXELS] {
    let mean = incoming.iter().sum::<f64>() / PIXELS as f64;
    incoming.map(|g| g - mean)
}
pub(crate) fn translate(image: &[f64; PIXELS], _dr: isize, _dc: isize) -> [f64; PIXELS] {
    // Canonical-view baseline: every requested training view is currently unchanged.
    *image
}
pub fn run(args: &[String]) -> Result<(), String> {
    residual::run(CORE, args)
}
pub fn check() -> Result<(), String> {
    residual::check(CORE)
}
