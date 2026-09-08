fn convolution_center(image: &[f64; 9], kernel: &[f64; 9]) -> f64 {
    image.iter().zip(kernel).map(|(x, w)| x * w).sum()
}
fn residual_add(_input: &[f64], _residual_branch: &[f64]) -> Vec<f64> {
    // TODO: add the branch to the matching input elements.
    todo!("return elementwise input plus branch")
}
fn main() {
    let _guided_step = residual_add;
    println!(
        "previous checkpoint convolution: {}",
        convolution_center(&[1.0; 9], &[0.5; 9])
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shortcut_adds_elementwise() {
        assert_eq!(residual_add(&[1.0, 2.0], &[0.5, -1.0]), [1.5, 1.0]);
    }
}
