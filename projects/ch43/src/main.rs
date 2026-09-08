//! Tiny decoder SFT, output-projection LoRA, and teacher/student distillation.
use ch36::{Config, Decoder, Gradients};
use std::{error::Error, ops::Range};

fn config(width: usize) -> Config {
    Config {
        vocab_size: 8,
        context: 4,
        width,
        heads: if width >= 8 { 2 } else { 1 },
        layers: 1,
        ff_width: width * 2,
    }
}
fn output_span(model: &Decoder) -> Range<usize> {
    let s = model
        .parameter_spans()
        .into_iter()
        .find(|s| s.name == "output_weight")
        .expect("ch36 output span");
    s.start..s.end
}
fn average_loss(model: &Decoder, data: &[([usize; 3], [usize; 3])]) -> Result<f32, Box<dyn Error>> {
    Ok(data
        .iter()
        .map(|(x, y)| model.loss_and_grad(x, y).map(|g| g.loss))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .sum::<f32>()
        / data.len() as f32)
}
fn sft(
    model: &mut Decoder,
    data: &[([usize; 3], [usize; 3])],
    steps: usize,
    rate: f32,
) -> Result<(), Box<dyn Error>> {
    for step in 0..steps {
        let (x, y) = &data[step % data.len()];
        let g = model.loss_and_grad(x, y)?;
        model.apply_sgd(&g, rate)?;
    }
    Ok(())
}

fn pretrained_base() -> Result<Decoder, Box<dyn Error>> {
    let mut model = Decoder::new(config(8), 43)?;
    let generic = [([0, 1, 2], [1, 2, 3]), ([4, 5, 6], [5, 6, 7])];
    sft(&mut model, &generic, 60, 0.06)?;
    Ok(model)
}

#[derive(Clone)]
struct Lora {
    a: Vec<f32>,
    b: Vec<f32>,
    rank: usize,
    alpha: f32,
    width: usize,
    vocab: usize,
}
impl Lora {
    fn new(width: usize, vocab: usize, rank: usize) -> Result<Self, &'static str> {
        if width == 0 || vocab == 0 || rank == 0 {
            return Err("LoRA dimensions and rank must be positive");
        }
        let a = (0..width * rank)
            .map(|i| ((i * 17 % 19) as f32 - 9.0) * 0.002)
            .collect();
        Ok(Self {
            a,
            b: vec![0.0; rank * vocab],
            rank,
            alpha: rank as f32,
            width,
            vocab,
        })
    }
    fn delta(&self) -> Vec<f32> {
        let scale = self.alpha / self.rank as f32;
        (0..self.width * self.vocab)
            .map(|ij| {
                let i = ij / self.vocab;
                let j = ij % self.vocab;
                (0..self.rank)
                    .map(|r| self.a[i * self.rank + r] * self.b[r * self.vocab + j])
                    .sum::<f32>()
                    * scale
            })
            .collect()
    }
    fn adapted(&self, base: &Decoder) -> Decoder {
        let mut model = base.clone();
        let span = output_span(&model);
        for (w, d) in model.parameters_mut()[span].iter_mut().zip(self.delta()) {
            *w += d;
        }
        model
    }
    fn step(
        &mut self,
        base: &Decoder,
        input: &[usize],
        targets: &[usize],
        rate: f32,
    ) -> Result<f32, Box<dyn Error>> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err("LoRA rate must be finite and positive".into());
        }
        let effective = self.adapted(base);
        let gradients = effective.loss_and_grad(input, targets)?;
        let dw = &gradients.values[output_span(&effective)];
        let scale = self.alpha / self.rank as f32;
        let mut da = vec![0.0; self.a.len()];
        let mut db = vec![0.0; self.b.len()];
        for i in 0..self.width {
            for r in 0..self.rank {
                da[i * self.rank + r] = scale
                    * (0..self.vocab)
                        .map(|j| dw[i * self.vocab + j] * self.b[r * self.vocab + j])
                        .sum::<f32>();
            }
        }
        for r in 0..self.rank {
            for j in 0..self.vocab {
                db[r * self.vocab + j] = scale
                    * (0..self.width)
                        .map(|i| self.a[i * self.rank + r] * dw[i * self.vocab + j])
                        .sum::<f32>();
            }
        }
        for (x, g) in self.a.iter_mut().zip(da) {
            *x -= rate * g;
        }
        for (x, g) in self.b.iter_mut().zip(db) {
            *x -= rate * g;
        }
        Ok(gradients.loss)
    }
}

fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let m = logits
        .iter()
        .map(|x| x / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|x| (x / temperature - m).exp()).collect();
    let z = p.iter().sum::<f32>();
    p.iter_mut().for_each(|x| *x /= z);
    p
}
fn last_logits(model: &Decoder, token: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    model.forward(&[token])
}
fn distill_gradient(
    student: &Decoder,
    token: usize,
    teacher_p: &[f32],
) -> Result<Gradients, Box<dyn Error>> {
    if teacher_p.len() != student.config().vocab_size
        || teacher_p.iter().any(|p| !p.is_finite() || *p < 0.0)
        || (teacher_p.iter().sum::<f32>() - 1.0).abs() > 1e-5
    {
        return Err("teacher probabilities must match the vocabulary and sum to one".into());
    }
    // ponytail: V hard-target backwards are transparent for V=8; add a soft-target backward for a larger vocabulary.
    let mut values = vec![0.0; student.parameter_count()];
    let mut loss = 0.0;
    for (class, &weight) in teacher_p.iter().enumerate() {
        let g = student.loss_and_grad(&[token], &[class])?;
        loss += weight * g.loss;
        for (dst, src) in values.iter_mut().zip(g.values) {
            *dst += weight * src;
        }
    }
    Ok(Gradients {
        loss,
        tokens: 1,
        values,
    })
}
fn log_softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<_> = logits.iter().map(|x| (x - maximum) / temperature).collect();
    let log_sum = shifted.iter().map(|x| x.exp()).sum::<f32>().ln();
    shifted.iter().map(|x| x - log_sum).collect()
}
fn kl_from_logits(teacher: &[f32], student: &[f32], temperature: f32) -> f32 {
    let teacher_log = log_softmax(teacher, temperature);
    let student_log = log_softmax(student, temperature);
    teacher_log
        .iter()
        .zip(student_log)
        .map(|(&log_p, log_q)| log_p.exp() * (log_p - log_q))
        .sum()
}
fn mean_teacher_kl(
    teacher: &Decoder,
    student: &Decoder,
    tokens: &[usize],
    temperature: f32,
) -> Result<f32, Box<dyn Error>> {
    let mut total = 0.0;
    for &t in tokens {
        total += kl_from_logits(
            &last_logits(teacher, t)?,
            &last_logits(student, t)?,
            temperature,
        );
    }
    Ok(total / tokens.len() as f32)
}

fn main() -> Result<(), Box<dyn Error>> {
    let data = [([1, 2, 3], [2, 3, 4]), ([5, 6, 2], [6, 2, 3])];
    let base = pretrained_base()?;
    let mut tuned = base.clone();
    let before = average_loss(&tuned, &data)?;
    sft(&mut tuned, &data, 120, 0.08)?;
    let after = average_loss(&tuned, &data)?;
    println!("SFT loss: {before:.4} -> {after:.4}");
    let mut lora = Lora::new(8, 8, 2)?;
    let unchanged = lora.adapted(&base);
    let zero_error = base
        .parameters()
        .iter()
        .zip(unchanged.parameters())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let lora_before = average_loss(&unchanged, &data)?;
    for step in 0..240 {
        let (x, y) = &data[step % data.len()];
        lora.step(&base, x, y, 0.4)?;
    }
    let adapted = lora.adapted(&base);
    let lora_after = average_loss(&adapted, &data)?;
    println!("LoRA zero-init max change={zero_error:.1}; loss {lora_before:.4} -> {lora_after:.4}; trained {} of {} base parameters",lora.a.len()+lora.b.len(),base.parameter_count());
    let mut student = Decoder::new(config(4), 7)?;
    let transfer = [1, 2, 3, 5];
    let kl_before = mean_teacher_kl(&tuned, &student, &transfer, 1.0)?;
    for step in 0..160 {
        let token = transfer[step % transfer.len()];
        let p = softmax(&last_logits(&tuned, token)?, 1.0);
        let g = distill_gradient(&student, token, &p)?;
        student.apply_sgd(&g, 0.12)?;
    }
    let kl_after = mean_teacher_kl(&tuned, &student, &transfer, 1.0)?;
    println!("distillation mean KL at T=1: {kl_before:.5} -> {kl_after:.5}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn soft_target_gradient_and_kl_are_consistent() {
        let student = Decoder::new(config(4), 7).unwrap();
        let teacher_p = [0.4, 0.1, 0.05, 0.15, 0.1, 0.05, 0.1, 0.05];
        let g = distill_gradient(&student, 1, &teacher_p).unwrap();
        let parameter = output_span(&student).start;
        let h = 1e-3;
        let mut plus = student.clone();
        let mut minus = student.clone();
        plus.parameters_mut()[parameter] += h;
        minus.parameters_mut()[parameter] -= h;
        let loss = |model: &Decoder| {
            teacher_p
                .iter()
                .zip(log_softmax(&last_logits(model, 1).unwrap(), 1.0))
                .map(|(&p, log_q)| -p * log_q)
                .sum::<f32>()
        };
        let numeric = (loss(&plus) - loss(&minus)) / (2.0 * h);
        assert!((numeric - g.values[parameter]).abs() < 3e-4 + 1e-3 * numeric.abs());
        assert!((kl_from_logits(&[0.0, -1000.0], &[-1000.0, 0.0], 1.0) - 1000.0).abs() < 1e-4);
        assert!(distill_gradient(&student, 1, &[0.5, 0.5]).is_err());
    }
    #[test]
    fn zero_initialized_adapter_preserves_base() {
        let b = pretrained_base().unwrap();
        let l = Lora::new(8, 8, 2).unwrap();
        assert_eq!(b.parameters(), l.adapted(&b).parameters());
    }
    #[test]
    fn sft_lora_and_distillation_improve_their_objectives() {
        let data = [([1, 2, 3], [2, 3, 4]), ([5, 6, 2], [6, 2, 3])];
        let base = pretrained_base().unwrap();
        let mut tuned = base.clone();
        let a = average_loss(&tuned, &data).unwrap();
        sft(&mut tuned, &data, 80, 0.08).unwrap();
        assert!(average_loss(&tuned, &data).unwrap() < a);
        let frozen = base.parameters().to_vec();
        let mut l = Lora::new(8, 8, 2).unwrap();
        let a = average_loss(&l.adapted(&base), &data).unwrap();
        for i in 0..160 {
            let (x, y) = &data[i % 2];
            l.step(&base, x, y, 0.4).unwrap();
        }
        assert!(average_loss(&l.adapted(&base), &data).unwrap() < a);
        assert_eq!(base.parameters(), frozen);
        let mut student = Decoder::new(config(4), 7).unwrap();
        let xs = [1, 2, 3];
        let a = mean_teacher_kl(&tuned, &student, &xs, 1.0).unwrap();
        for i in 0..120 {
            let t = xs[i % xs.len()];
            let p = softmax(&last_logits(&tuned, t).unwrap(), 1.0);
            let g = distill_gradient(&student, t, &p).unwrap();
            student.apply_sgd(&g, 0.12).unwrap();
        }
        assert!(mean_teacher_kl(&tuned, &student, &xs, 1.0).unwrap() < a);
    }
    #[test]
    fn adapter_gradient_matches_finite_difference() {
        let base = pretrained_base().unwrap();
        let mut l = Lora::new(8, 8, 2).unwrap();
        l.b[0] = 0.1;
        let h = 1e-3;
        let mut plus = l.clone();
        plus.a[0] += h;
        let mut minus = l.clone();
        minus.a[0] -= h;
        let loss = |x: &Lora| {
            x.adapted(&base)
                .loss_and_grad(&[1, 2], &[2, 3])
                .unwrap()
                .loss
        };
        let numeric = (loss(&plus) - loss(&minus)) / (2.0 * h);
        let old = l.a[0];
        l.step(&base, &[1, 2], &[2, 3], 1e-3).unwrap();
        let analytic = (old - l.a[0]) / 1e-3;
        assert!((numeric - analytic).abs() < 2e-3, "{numeric} vs {analytic}");
        assert!(Lora::new(8, 8, 0).is_err());
    }
}
