//! Two trainable pre-normalized residual image blocks with deterministic augmentation.
const SIDE: usize = 6;
const PIXELS: usize = SIDE * SIDE;
const CLASSES: usize = 2;
const BLOCKS: usize = 2;

#[derive(Clone)]
struct Example {
    image: [f64; PIXELS],
    label: usize,
}

fn normalize(input: &[f64; PIXELS]) -> ([f64; PIXELS], f64) {
    let mean = input.iter().sum::<f64>() / PIXELS as f64;
    let variance = input
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / PIXELS as f64;
    let inv_std = 1.0 / (variance + 1e-5).sqrt();
    (input.map(|value| (value - mean) * inv_std), inv_std)
}

fn normalize_backward(
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

fn cross_entropy_from_logits(logits: [f64; CLASSES], target: usize) -> f64 {
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (maximum - logits[target])
        + logits
            .iter()
            .map(|logit| (logit - maximum).exp())
            .sum::<f64>()
            .ln()
}

fn translate(image: &[f64; PIXELS], dr: isize, dc: isize) -> [f64; PIXELS] {
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

fn base(label: usize, thickness: usize) -> [f64; PIXELS] {
    let mut x = [0.05; PIXELS];
    for t in 0..thickness {
        for k in 1..SIDE - 1 {
            let (r, c) = if label == 0 { (k, 2 + t) } else { (2 + t, k) };
            x[r * SIDE + c] = 1.0;
        }
    }
    x
}

fn data(train: bool) -> Vec<Example> {
    let shifts: &[(isize, isize)] = if train {
        &[(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
    } else {
        &[(0, 0)]
    };
    let mut examples = Vec::new();
    for label in 0..CLASSES {
        for thickness in 1..=2 {
            for &(dr, dc) in shifts {
                let mut image = translate(&base(label, thickness), dr, dc);
                if !train {
                    for value in &mut image {
                        *value *= 0.9;
                    }
                    image[(label * 13 + thickness * 7) % PIXELS] += 0.03;
                }
                examples.push(Example { image, label });
            }
        }
    }
    examples
}

#[derive(Clone)]
struct ResidualNet {
    kernels: [[f64; 9]; BLOCKS],
    scales: [f64; BLOCKS],
    branch_biases: [f64; BLOCKS],
    head: Vec<f64>,
    class_bias: [f64; CLASSES],
}

struct ResidualBlockCache {
    normalized: [f64; PIXELS],
    inv_std: f64,
    branch: [f64; PIXELS],
    output: [f64; PIXELS],
}
struct ForwardCache {
    blocks: [ResidualBlockCache; BLOCKS],
    logits: [f64; CLASSES],
}
struct Gradient {
    kernels: [[f64; 9]; BLOCKS],
    scales: [f64; BLOCKS],
    branch_biases: [f64; BLOCKS],
    head: Vec<f64>,
    class_bias: [f64; CLASSES],
}

impl ResidualNet {
    fn new() -> Self {
        let mut head = vec![0.0; CLASSES * PIXELS];
        for (i, w) in head.iter_mut().enumerate() {
            *w = ((i * 29 + 3) as f64).sin() * 0.03;
        }
        Self {
            kernels: [
                [-0.05, 0.02, 0.05, -0.08, 0.02, 0.08, -0.05, 0.02, 0.05],
                [0.03, -0.04, 0.02, 0.05, 0.01, -0.06, 0.02, -0.03, 0.04],
            ],
            scales: [0.1, 0.1],
            branch_biases: [0.02, 0.01],
            head,
            class_bias: [0.0; CLASSES],
        }
    }

    fn residual_block_forward(
        input: &[f64; PIXELS],
        kernel: &[f64; 9],
        scale: f64,
        bias: f64,
    ) -> ResidualBlockCache {
        let (normalized, inv_std) = normalize(input);
        let mut branch = [0.0; PIXELS];
        let mut output = [0.0; PIXELS];
        for r in 0..SIDE {
            for c in 0..SIDE {
                let mut z = bias;
                for kr in 0..3 {
                    for kc in 0..3 {
                        let ir = r as isize + kr as isize - 1;
                        let ic = c as isize + kc as isize - 1;
                        if (0..SIDE as isize).contains(&ir) && (0..SIDE as isize).contains(&ic) {
                            z += kernel[kr * 3 + kc] * normalized[ir as usize * SIDE + ic as usize];
                        }
                    }
                }
                let at = r * SIDE + c;
                branch[at] = z;
                output[at] = (input[at] + scale * z).max(0.0);
            }
        }
        ResidualBlockCache {
            normalized,
            inv_std,
            branch,
            output,
        }
    }

    fn forward(&self, image: &[f64; PIXELS]) -> ForwardCache {
        let first = Self::residual_block_forward(
            image,
            &self.kernels[0],
            self.scales[0],
            self.branch_biases[0],
        );
        let second = Self::residual_block_forward(
            &first.output,
            &self.kernels[1],
            self.scales[1],
            self.branch_biases[1],
        );
        let mut logits = self.class_bias;
        for (class, logit) in logits.iter_mut().enumerate() {
            for j in 0..PIXELS {
                *logit += self.head[class * PIXELS + j] * second.output[j];
            }
        }
        ForwardCache {
            blocks: [first, second],
            logits,
        }
    }

    fn residual_block_backward(
        cache: &ResidualBlockCache,
        kernel: &[f64; 9],
        scale: f64,
        output_gradient: &[f64; PIXELS],
    ) -> ([f64; PIXELS], [f64; 9], f64, f64) {
        let mut input_gradient = [0.0; PIXELS];
        let mut normalized_gradient = [0.0; PIXELS];
        let mut kernel_gradient = [0.0; 9];
        let mut scale_gradient = 0.0;
        let mut bias_gradient = 0.0;
        for r in 0..SIDE {
            for c in 0..SIDE {
                let at = r * SIDE + c;
                if cache.output[at] <= 0.0 {
                    continue;
                }
                let pre_activation_gradient = output_gradient[at];
                input_gradient[at] += pre_activation_gradient;
                scale_gradient += pre_activation_gradient * cache.branch[at];
                let branch_gradient = pre_activation_gradient * scale;
                bias_gradient += branch_gradient;
                for kr in 0..3 {
                    for kc in 0..3 {
                        let ir = r as isize + kr as isize - 1;
                        let ic = c as isize + kc as isize - 1;
                        if (0..SIDE as isize).contains(&ir) && (0..SIDE as isize).contains(&ic) {
                            let source = ir as usize * SIDE + ic as usize;
                            kernel_gradient[kr * 3 + kc] +=
                                branch_gradient * cache.normalized[source];
                            normalized_gradient[source] += branch_gradient * kernel[kr * 3 + kc];
                        }
                    }
                }
            }
        }
        let through_normalization =
            normalize_backward(&normalized_gradient, &cache.normalized, cache.inv_std);
        for j in 0..PIXELS {
            input_gradient[j] += through_normalization[j];
        }
        (
            input_gradient,
            kernel_gradient,
            scale_gradient,
            bias_gradient,
        )
    }

    fn loss_and_gradient(&self, example: &Example) -> (f64, Gradient) {
        let cache = self.forward(&example.image);
        let loss = cross_entropy_from_logits(cache.logits, example.label);
        let maximum = cache
            .logits
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let exponential_sum: f64 = cache
            .logits
            .iter()
            .map(|logit| (logit - maximum).exp())
            .sum();
        let mut logit_gradient = cache
            .logits
            .map(|logit| (logit - maximum).exp() / exponential_sum);
        logit_gradient[example.label] -= 1.0;
        let mut gradient = Gradient {
            kernels: [[0.0; 9]; BLOCKS],
            scales: [0.0; BLOCKS],
            branch_biases: [0.0; BLOCKS],
            head: vec![0.0; self.head.len()],
            class_bias: logit_gradient,
        };
        let mut second_block_output_gradient = [0.0; PIXELS];
        for (class, &class_gradient) in logit_gradient.iter().enumerate() {
            for (j, output_gradient) in second_block_output_gradient.iter_mut().enumerate() {
                gradient.head[class * PIXELS + j] = class_gradient * cache.blocks[1].output[j];
                *output_gradient += self.head[class * PIXELS + j] * class_gradient;
            }
        }
        let (first_block_output_gradient, kernel2, scale2, bias2) = Self::residual_block_backward(
            &cache.blocks[1],
            &self.kernels[1],
            self.scales[1],
            &second_block_output_gradient,
        );
        let (_, kernel1, scale1, bias1) = Self::residual_block_backward(
            &cache.blocks[0],
            &self.kernels[0],
            self.scales[0],
            &first_block_output_gradient,
        );
        gradient.kernels = [kernel1, kernel2];
        gradient.scales = [scale1, scale2];
        gradient.branch_biases = [bias1, bias2];
        (loss, gradient)
    }

    fn train(
        &mut self,
        data: &[Example],
        epochs: usize,
        learning_rate: f64,
    ) -> Result<(), &'static str> {
        if data.is_empty() || epochs == 0 || !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("nonempty data, positive epochs and learning rate required");
        }
        if data.iter().any(|example| {
            example.label >= CLASSES || example.image.iter().any(|value| !value.is_finite())
        }) {
            return Err("images must be finite and labels in range");
        }
        for _ in 0..epochs {
            for example in data {
                let (loss, gradient) = self.loss_and_gradient(example);
                if !loss.is_finite() {
                    return Err("non-finite training loss; reduce the learning rate");
                }
                for block in 0..BLOCKS {
                    for k in 0..9 {
                        self.kernels[block][k] -= learning_rate * gradient.kernels[block][k];
                    }
                    self.scales[block] -= learning_rate * gradient.scales[block];
                    self.branch_biases[block] -= learning_rate * gradient.branch_biases[block];
                }
                for (weight, weight_gradient) in self.head.iter_mut().zip(gradient.head) {
                    *weight -= learning_rate * weight_gradient;
                }
                for class in 0..CLASSES {
                    self.class_bias[class] -= learning_rate * gradient.class_bias[class];
                }
                if self
                    .kernels
                    .iter()
                    .flatten()
                    .chain(&self.scales)
                    .chain(&self.branch_biases)
                    .chain(&self.head)
                    .chain(&self.class_bias)
                    .any(|x| !x.is_finite())
                {
                    return Err("training update overflow; reduce the learning rate");
                }
            }
        }
        Ok(())
    }

    fn metrics(&self, data: &[Example]) -> (f64, f64) {
        let mut loss = 0.0;
        let mut correct = 0;
        for example in data {
            let cache = self.forward(&example.image);
            loss += cross_entropy_from_logits(cache.logits, example.label);
            let prediction = usize::from(cache.logits[1] > cache.logits[0]);
            correct += usize::from(prediction == example.label);
        }
        (loss / data.len() as f64, correct as f64 / data.len() as f64)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_data = data(true);
    let validation_data = data(false);
    let mut model = ResidualNet::new();
    let before = model.metrics(&validation_data);
    model.train(&train_data, 35, 0.015)?;
    let after = model.metrics(&validation_data);
    println!(
        "augmented training images={}, disjoint validation images={}",
        train_data.len(),
        validation_data.len()
    );
    println!(
        "validation cross-entropy {:.4} -> {:.4}; accuracy {:.1}% -> {:.1}%",
        before.0,
        after.0,
        before.1 * 100.0,
        after.1 * 100.0
    );
    println!(
        "learned residual scales [{:.4}, {:.4}]",
        model.scales[0], model.scales[1]
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loss_is_stable_at_large_common_bias_and_bad_images_are_rejected() {
        let mut model = ResidualNet::new();
        model.head.fill(0.0);
        model.class_bias.fill(1e16);
        let mut examples = data(true);
        assert!((model.loss_and_gradient(&examples[0]).0 - 2.0_f64.ln()).abs() < 1e-12);
        assert!((model.metrics(&examples[..1]).0 - 2.0_f64.ln()).abs() < 1e-12);
        examples[0].image[0] = f64::NAN;
        assert!(model.train(&examples, 1, 0.1).is_err());
    }

    #[test]
    fn first_block_gradient_matches_central_difference_through_second_block() {
        let mut model = ResidualNet::new();
        let example = data(false).remove(0);
        let analytic = model.loss_and_gradient(&example).1.kernels[0][0];
        let h = 1e-5;
        model.kernels[0][0] += h;
        let plus = model.loss_and_gradient(&example).0;
        model.kernels[0][0] -= 2.0 * h;
        let minus = model.loss_and_gradient(&example).0;
        let numeric = (plus - minus) / (2.0 * h);
        assert!(
            (analytic - numeric).abs() < 1e-6 + 1e-4 * numeric.abs(),
            "{analytic} {numeric}"
        );
    }
    #[test]
    fn training_improves_disjoint_validation_loss() {
        let mut model = ResidualNet::new();
        let validation_data = data(false);
        let before = model.metrics(&validation_data).0;
        model.train(&data(true), 35, 0.015).unwrap();
        let after = model.metrics(&validation_data);
        assert!(
            after.0 < before * 0.5 && after.1 == 1.0,
            "{before} {after:?}"
        );
    }
    #[test]
    fn normalization_and_split_invariants_hold() {
        let (n, _) = normalize(&base(0, 1));
        assert!(n.iter().sum::<f64>().abs() < 1e-10);
        let train_data = data(true);
        let validation_data = data(false);
        assert!(validation_data.iter().all(|validation_example| !train_data
            .iter()
            .any(|train_example| train_example.image == validation_example.image)));
    }
}
