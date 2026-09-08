const SIDE: usize = 4;
const IMAGES: [[f64; SIDE * SIDE]; 2] = [
    [
        0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
    ],
    [
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    ],
];

fn stable_softmax(logits: [f64; 2]) -> [f64; 2] {
    let m = logits[0].max(logits[1]);
    let exp = [(logits[0] - m).exp(), (logits[1] - m).exp()];
    let sum = exp[0] + exp[1];
    [exp[0] / sum, exp[1] / sum]
}

fn linear_logits(image: &[f64; SIDE * SIDE]) -> [f64; 2] {
    let vertical: f64 = (0..SIDE)
        .map(|r| image[r * SIDE + 1] + image[r * SIDE + 2])
        .sum();
    let horizontal: f64 = image[SIDE..2 * SIDE]
        .iter()
        .chain(&image[2 * SIDE..3 * SIDE])
        .sum();
    [vertical - horizontal, horizontal - vertical]
}

fn convolution_at(
    _image: &[f64],
    _side: usize,
    _kernel: &[f64; 9],
    _row: usize,
    _col: usize,
) -> f64 {
    // TODO: apply a padded 3x3 cross-correlation at this output position.
    todo!("sum valid image pixels times their shared kernel weights")
}

fn main() {
    let _guided_step = convolution_at;
    println!("fixture: {} images, shape {}x{}", IMAGES.len(), SIDE, SIDE);
    for image in &IMAGES {
        let logits = linear_logits(image);
        println!(
            "linear logits {logits:?}, probabilities {:?}",
            stable_softmax(logits)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn center_convolution_matches_hand_sum() {
        let image: Vec<f64> = (1..=9).map(|v| v as f64).collect();
        assert_eq!(convolution_at(&image, 3, &[1.0; 9], 1, 1), 45.0);
        assert_eq!(convolution_at(&image, 3, &[1.0; 9], 0, 0), 12.0);
        let kernel = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(convolution_at(&image, 3, &kernel, 1, 1), 5.0);
    }
}
