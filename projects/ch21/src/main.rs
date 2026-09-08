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

fn normalize(image: &[f64; PIXELS]) -> ([f64; PIXELS], f64) {
    let mean = image.iter().sum::<f64>() / PIXELS as f64;
    let variance = image.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / PIXELS as f64;
    let inv_std = 1.0 / (variance + 1e-5).sqrt();
    (image.map(|x| (x - mean) * inv_std), inv_std)
}

fn normalize_backward(dy: &[f64; PIXELS], y: &[f64; PIXELS], inv_std: f64) -> [f64; PIXELS] {
    let sum_dy: f64 = dy.iter().sum();
    let sum_dy_y: f64 = dy.iter().zip(y).map(|(d, n)| d * n).sum();
    std::array::from_fn(|j| {
        inv_std * (PIXELS as f64 * dy[j] - sum_dy - y[j] * sum_dy_y) / PIXELS as f64
    })
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
    let mut rows = Vec::new();
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
                rows.push(Example { image, label });
            }
        }
    }
    rows
}

#[derive(Clone)]
struct ResidualNet {
    kernels: [[f64; 9]; BLOCKS],
    scales: [f64; BLOCKS],
    branch_biases: [f64; BLOCKS],
    head: Vec<f64>,
    class_bias: [f64; CLASSES],
}

struct BlockCache {
    norm: [f64; PIXELS],
    inv_std: f64,
    branch: [f64; PIXELS],
    output: [f64; PIXELS],
}
struct NetCache {
    blocks: [BlockCache; BLOCKS],
    logits: [f64; CLASSES],
}
struct Grad {
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

    fn block(input: &[f64; PIXELS], kernel: &[f64; 9], scale: f64, bias: f64) -> BlockCache {
        let (norm, inv_std) = normalize(input);
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
                            z += kernel[kr * 3 + kc] * norm[ir as usize * SIDE + ic as usize];
                        }
                    }
                }
                let at = r * SIDE + c;
                branch[at] = z;
                output[at] = (input[at] + scale * z).max(0.0);
            }
        }
        BlockCache {
            norm,
            inv_std,
            branch,
            output,
        }
    }

    fn forward(&self, image: &[f64; PIXELS]) -> NetCache {
        let first = Self::block(
            image,
            &self.kernels[0],
            self.scales[0],
            self.branch_biases[0],
        );
        let second = Self::block(
            &first.output,
            &self.kernels[1],
            self.scales[1],
            self.branch_biases[1],
        );
        let mut logits = self.class_bias;
        for (y, logit) in logits.iter_mut().enumerate() {
            for j in 0..PIXELS {
                *logit += self.head[y * PIXELS + j] * second.output[j];
            }
        }
        NetCache {
            blocks: [first, second],
            logits,
        }
    }

    fn block_backward(
        cache: &BlockCache,
        kernel: &[f64; 9],
        scale: f64,
        dout: &[f64; PIXELS],
    ) -> ([f64; PIXELS], [f64; 9], f64, f64) {
        let mut dinput = [0.0; PIXELS];
        let mut dnorm = [0.0; PIXELS];
        let mut dkernel = [0.0; 9];
        let mut dscale = 0.0;
        let mut dbias = 0.0;
        for r in 0..SIDE {
            for c in 0..SIDE {
                let at = r * SIDE + c;
                if cache.output[at] <= 0.0 {
                    continue;
                }
                let dpre = dout[at];
                dinput[at] += dpre;
                dscale += dpre * cache.branch[at];
                let dbranch = dpre * scale;
                dbias += dbranch;
                for kr in 0..3 {
                    for kc in 0..3 {
                        let ir = r as isize + kr as isize - 1;
                        let ic = c as isize + kc as isize - 1;
                        if (0..SIDE as isize).contains(&ir) && (0..SIDE as isize).contains(&ic) {
                            let source = ir as usize * SIDE + ic as usize;
                            dkernel[kr * 3 + kc] += dbranch * cache.norm[source];
                            dnorm[source] += dbranch * kernel[kr * 3 + kc];
                        }
                    }
                }
            }
        }
        let through_norm = normalize_backward(&dnorm, &cache.norm, cache.inv_std);
        for j in 0..PIXELS {
            dinput[j] += through_norm[j];
        }
        (dinput, dkernel, dscale, dbias)
    }

    fn loss_grad(&self, row: &Example) -> (f64, Grad) {
        let c = self.forward(&row.image);
        let m = c.logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = c.logits.iter().map(|&z| (z - m).exp()).sum();
        let loss = (m - c.logits[row.label]) + sum.ln();
        let mut dz = c.logits.map(|z| (z - m).exp() / sum);
        dz[row.label] -= 1.0;
        let mut g = Grad {
            kernels: [[0.0; 9]; BLOCKS],
            scales: [0.0; BLOCKS],
            branch_biases: [0.0; BLOCKS],
            head: vec![0.0; self.head.len()],
            class_bias: dz,
        };
        let mut dout = [0.0; PIXELS];
        for (y, &dy) in dz.iter().enumerate() {
            for (j, dout_value) in dout.iter_mut().enumerate() {
                g.head[y * PIXELS + j] = dy * c.blocks[1].output[j];
                *dout_value += self.head[y * PIXELS + j] * dy;
            }
        }
        let (d_first, k2, s2, b2) =
            Self::block_backward(&c.blocks[1], &self.kernels[1], self.scales[1], &dout);
        let (_, k1, s1, b1) =
            Self::block_backward(&c.blocks[0], &self.kernels[0], self.scales[0], &d_first);
        g.kernels = [k1, k2];
        g.scales = [s1, s2];
        g.branch_biases = [b1, b2];
        (loss, g)
    }

    fn train(&mut self, rows: &[Example], epochs: usize, rate: f64) -> Result<(), &'static str> {
        if rows.is_empty() || epochs == 0 || !rate.is_finite() || rate <= 0.0 {
            return Err("nonempty data, positive epochs and rate required");
        }
        if rows
            .iter()
            .any(|r| r.label >= CLASSES || r.image.iter().any(|x| !x.is_finite()))
        {
            return Err("images must be finite and labels in range");
        }
        for _ in 0..epochs {
            for row in rows {
                let (loss, g) = self.loss_grad(row);
                if !loss.is_finite() {
                    return Err("non-finite training loss; reduce the rate");
                }
                for block in 0..BLOCKS {
                    for k in 0..9 {
                        self.kernels[block][k] -= rate * g.kernels[block][k];
                    }
                    self.scales[block] -= rate * g.scales[block];
                    self.branch_biases[block] -= rate * g.branch_biases[block];
                }
                for (w, dw) in self.head.iter_mut().zip(g.head) {
                    *w -= rate * dw;
                }
                for y in 0..CLASSES {
                    self.class_bias[y] -= rate * g.class_bias[y];
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

    fn metrics(&self, rows: &[Example]) -> (f64, f64) {
        let mut loss = 0.0;
        let mut correct = 0;
        for row in rows {
            let c = self.forward(&row.image);
            let m = c.logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            loss += (m - c.logits[row.label])
                + c.logits.iter().map(|&z| (z - m).exp()).sum::<f64>().ln();
            correct += usize::from(usize::from(c.logits[1] > c.logits[0]) == row.label);
        }
        (loss / rows.len() as f64, correct as f64 / rows.len() as f64)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train = data(true);
    let valid = data(false);
    let mut model = ResidualNet::new();
    let before = model.metrics(&valid);
    model.train(&train, 35, 0.015)?;
    let after = model.metrics(&valid);
    println!(
        "augmented training images={}, disjoint validation images={}",
        train.len(),
        valid.len()
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
        let mut rows = data(true);
        assert!((model.loss_grad(&rows[0]).0 - 2.0_f64.ln()).abs() < 1e-12);
        assert!((model.metrics(&rows[..1]).0 - 2.0_f64.ln()).abs() < 1e-12);
        rows[0].image[0] = f64::NAN;
        assert!(model.train(&rows, 1, 0.1).is_err());
    }

    #[test]
    fn first_block_gradient_matches_central_difference_through_second_block() {
        let mut model = ResidualNet::new();
        let row = data(false).remove(0);
        let analytic = model.loss_grad(&row).1.kernels[0][0];
        let h = 1e-5;
        model.kernels[0][0] += h;
        let plus = model.loss_grad(&row).0;
        model.kernels[0][0] -= 2.0 * h;
        let minus = model.loss_grad(&row).0;
        let numeric = (plus - minus) / (2.0 * h);
        assert!(
            (analytic - numeric).abs() < 1e-6 + 1e-4 * numeric.abs(),
            "{analytic} {numeric}"
        );
    }
    #[test]
    fn training_improves_disjoint_validation_loss() {
        let mut model = ResidualNet::new();
        let valid = data(false);
        let before = model.metrics(&valid).0;
        model.train(&data(true), 35, 0.015).unwrap();
        let after = model.metrics(&valid);
        assert!(
            after.0 < before * 0.5 && after.1 == 1.0,
            "{before} {after:?}"
        );
    }
    #[test]
    fn normalization_and_split_invariants_hold() {
        let (n, _) = normalize(&base(0, 1));
        assert!(n.iter().sum::<f64>().abs() < 1e-10);
        let train = data(true);
        let valid = data(false);
        assert!(valid
            .iter()
            .all(|v| !train.iter().any(|t| t.image == v.image)));
    }
}
