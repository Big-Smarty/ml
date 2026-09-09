//! B is prepared once as weights-transpose [k,n]. Initialize bias once, then accumulate every inner tile.
use crate::common::{self, Shape};
pub fn blocked(
    x: &[f32],
    b: &[f32],
    bias: &[f32],
    y: &mut [f32],
    s: Shape,
    block: usize,
) -> Result<usize, String> {
    common::validate(x, b, bias, y, s)?;
    if block == 0 {
        return Err("block must be positive".into());
    }
    for row in y.chunks_exact_mut(s.n) {
        row.copy_from_slice(bias);
    }
    let mut visited = 0;
    // ponytail: one tile level; packing/register blocking waits for measured need.
    for row_start in (0..s.m).step_by(block) {
        for inner_start in (0..s.k).step_by(block) {
            for col_start in (0..s.n).step_by(block) {
                visited += 1;
                for row in row_start..row_start.saturating_add(block).min(s.m) {
                    for inner in inner_start..inner_start.saturating_add(block).min(s.k) {
                        let value = x[row * s.k + inner];
                        for col in col_start..col_start.saturating_add(block).min(s.n) {
                            y[row * s.n + col] += value * b[inner * s.n + col];
                        }
                    }
                }
            }
        }
    }
    common::finite(y)?;
    Ok(visited)
}
pub fn dense(x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32], s: Shape) -> Result<(), String> {
    common::validate(x, w, bias, y, s)?;
    let b = crate::ch26::transpose(w, s);
    blocked(x, &b, bias, y, s, 4)?;
    Ok(())
}
pub fn run(args: &[String]) -> Result<(), String> {
    common::demo(dense)?;
    if args.iter().any(|a| a == "--bench") {
        crate::ch26::experiment(blocked)?;
    }
    Ok(())
}
pub fn check() -> Result<(), String> {
    crate::ch26::verify(blocked)
}
