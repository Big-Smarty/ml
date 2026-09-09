//! Completed spatial core. The kernel is never flipped: this is cross-correlation.
use crate::vision::{self, Core};
pub(crate) const CORE: Core = Core {
    convolve,
    pool,
    backward,
};

pub(crate) fn convolve(
    image: &[f64],
    rows: usize,
    cols: usize,
    kernel: &[f64; 9],
    bias: f64,
) -> Vec<f64> {
    let mut output = vec![bias; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            for kr in 0..3 {
                for kc in 0..3 {
                    let ir = r as isize + kr as isize - 1;
                    let ic = c as isize + kc as isize - 1;
                    if (0..rows as isize).contains(&ir) && (0..cols as isize).contains(&ic) {
                        output[r * cols + c] +=
                            kernel[kr * 3 + kc] * image[ir as usize * cols + ic as usize];
                    }
                }
            }
        }
    }
    output
}

pub(crate) fn pool(values: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<usize>) {
    let mut output = Vec::new();
    let mut winners = Vec::new();
    for r in 0..rows / 2 {
        for c in 0..cols / 2 {
            let first = 2 * r * cols + 2 * c;
            let mut winner = first;
            // Strict > keeps the first row-major winner on a tie.
            for at in [first + 1, first + cols, first + cols + 1] {
                if values[at] > values[winner] {
                    winner = at;
                }
            }
            output.push(values[winner]);
            winners.push(winner);
        }
    }
    (output, winners)
}

pub(crate) fn backward(image: &[f64], rows: usize, cols: usize, incoming: &[f64]) -> [f64; 9] {
    let mut gradient = [0.0; 9];
    for r in 0..rows {
        for c in 0..cols {
            for kr in 0..3 {
                for kc in 0..3 {
                    let ir = r as isize + kr as isize - 1;
                    let ic = c as isize + kc as isize - 1;
                    if (0..rows as isize).contains(&ir) && (0..cols as isize).contains(&ic) {
                        // One shared weight collects contributions from every output position.
                        gradient[kr * 3 + kc] +=
                            incoming[r * cols + c] * image[ir as usize * cols + ic as usize];
                    }
                }
            }
        }
    }
    gradient
}
pub fn run(args: &[String]) -> Result<(), String> {
    vision::run(CORE, args)
}
pub fn check() -> Result<(), String> {
    vision::check(CORE)
}
