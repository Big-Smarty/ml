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
    if data.is_empty() {
        return Err("loss requires nonempty data".into());
    }
    Ok(data
        .iter()
        .map(|(input, targets)| model.loss(input, targets))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .sum::<f32>()
        / data.len() as f32)
}
fn pretrain(
    model: &mut Decoder,
    data: &[([usize; 3], [usize; 3])],
    steps: usize,
    learning_rate: f32,
) -> Result<(), Box<dyn Error>> {
    if data.is_empty() {
        return Err("training requires examples".into());
    }
    for step in 0..steps {
        let (input, targets) = &data[step % data.len()];
        let gradients = model.loss_and_gradient(input, targets)?;
        model.apply_sgd(&gradients, learning_rate)?;
    }
    Ok(())
}

/// Prefix loss sums telescope: t*L(prefix_t) - (t-1)*L(prefix_{t-1})
/// is the loss/gradient of the last target, with its full causal context.
// ponytail: repeated prefix backwards cost O(T) full passes; expose masked dlogits in the decoder for large contexts.
fn masked_gradient(
    model: &Decoder,
    input: &[usize],
    targets: &[usize],
    mask: &[bool],
) -> Result<Gradients, Box<dyn Error>> {
    if input.is_empty()
        || input.len() != targets.len()
        || input.len() != mask.len()
        || !mask.iter().any(|x| *x)
    {
        return Err("mask needs aligned nonempty inputs and a supervised response".into());
    }
    let count = mask.iter().filter(|x| **x).count();
    let mut result = Gradients {
        loss: 0.,
        tokens: count,
        values: vec![0.; model.parameter_count()],
    };
    for (i, &supervised) in mask.iter().enumerate() {
        if !supervised {
            continue;
        }
        let current = model.loss_and_gradient(&input[..=i], &targets[..=i])?;
        let factor = (i + 1) as f32 / count as f32;
        result.loss += factor * current.loss;
        for (a, b) in result.values.iter_mut().zip(current.values) {
            *a += factor * b;
        }
        if i > 0 {
            let previous = model.loss_and_gradient(&input[..i], &targets[..i])?;
            let factor = i as f32 / count as f32;
            result.loss -= factor * previous.loss;
            for (a, b) in result.values.iter_mut().zip(previous.values) {
                *a -= factor * b;
            }
        }
    }
    Ok(result)
}
fn sft(
    model: &mut Decoder,
    data: &[([usize; 3], [usize; 3])],
    steps: usize,
    learning_rate: f32,
) -> Result<(), Box<dyn Error>> {
    if data.is_empty() {
        return Err("SFT needs demonstrations".into());
    }
    for step in 0..steps {
        let (x, y) = &data[step % data.len()];
        let g = masked_gradient(model, x, y, &[false, false, true])?;
        model.apply_sgd(&g, learning_rate)?;
    }
    Ok(())
}

fn pretrained_base() -> Result<Decoder, Box<dyn Error>> {
    let mut model = Decoder::new(config(8), 43)?;
    let generic = [([0, 1, 2], [1, 2, 3]), ([4, 5, 6], [5, 6, 7])];
    pretrain(&mut model, &generic, 60, 0.06)?;
    Ok(model)
}

fn low_rank_delta(
    a: &[f32],
    b: &[f32],
    input_features: usize,
    rank: usize,
    output_features: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), input_features * rank);
    assert_eq!(b.len(), rank * output_features);
    (0..input_features * output_features)
        .map(|index| {
            let row = index / output_features;
            let col = index % output_features;
            (0..rank)
                .map(|inner| a[row * rank + inner] * b[inner * output_features + col])
                .sum::<f32>()
        })
        .collect()
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
        let a_len = width.checked_mul(rank).ok_or("LoRA shape overflow")?;
        let b_len = rank.checked_mul(vocab).ok_or("LoRA shape overflow")?;
        let a = (0..a_len)
            .map(|i| ((i * 17 % 19) as f32 - 9.0) * 0.002)
            .collect();
        Ok(Self {
            a,
            b: vec![0.0; b_len],
            rank,
            alpha: rank as f32,
            width,
            vocab,
        })
    }
    fn delta(&self) -> Vec<f32> {
        let scale = self.alpha / self.rank as f32;
        low_rank_delta(&self.a, &self.b, self.width, self.rank, self.vocab)
            .into_iter()
            .map(|value| value * scale)
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
    /// Direct unmerged adapter path for one hidden row; compare it with merged weights.
    fn project(&self, hidden: &[f32], base: &Decoder) -> Result<Vec<f32>, Box<dyn Error>> {
        if hidden.len() != self.width
            || hidden.iter().any(|x| !x.is_finite())
            || base.config().width != self.width
            || base.config().vocab_size != self.vocab
        {
            return Err("adapter projection shape or value mismatch".into());
        }
        let weights = &base.parameters()[output_span(base)];
        let low: Vec<f32> = (0..self.rank)
            .map(|r| {
                (0..self.width)
                    .map(|i| hidden[i] * self.a[i * self.rank + r])
                    .sum()
            })
            .collect();
        let output: Vec<f32> = (0..self.vocab)
            .map(|j| {
                let base_value: f32 = (0..self.width)
                    .map(|i| hidden[i] * weights[i * self.vocab + j])
                    .sum();
                let adapter: f32 = (0..self.rank)
                    .map(|r| low[r] * self.b[r * self.vocab + j])
                    .sum();
                base_value + self.alpha / self.rank as f32 * adapter
            })
            .collect();
        if output.iter().any(|x| !x.is_finite()) {
            return Err("adapter projection overflow".into());
        }
        Ok(output)
    }
    fn step(
        &mut self,
        base: &Decoder,
        input: &[usize],
        targets: &[usize],
        learning_rate: f32,
    ) -> Result<f32, Box<dyn Error>> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("LoRA learning rate must be finite and positive".into());
        }
        let effective = self.adapted(base);
        let gradients = effective.loss_and_gradient(input, targets)?;
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
            *x -= learning_rate * g;
        }
        for (x, g) in self.b.iter_mut().zip(db) {
            *x -= learning_rate * g;
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
    teacher_probabilities: &[f32],
) -> Result<Gradients, Box<dyn Error>> {
    if teacher_probabilities.len() != student.config().vocab_size
        || teacher_probabilities
            .iter()
            .any(|probability| !probability.is_finite() || *probability < 0.0)
        || (teacher_probabilities.iter().sum::<f32>() - 1.0).abs() > 1e-5
    {
        return Err("teacher probabilities must match the vocabulary and sum to one".into());
    }
    // ponytail: V hard-target backwards are transparent for V=8; add a soft-target backward for a larger vocabulary.
    let mut values = vec![0.0; student.parameter_count()];
    let mut loss = 0.0;
    for (target, &weight) in teacher_probabilities.iter().enumerate() {
        let gradients = student.loss_and_gradient(&[token], &[target])?;
        loss += weight * gradients.loss;
        for (dst, src) in values.iter_mut().zip(gradients.values) {
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

fn experiment() -> Result<(), Box<dyn Error>> {
    let data = [([1, 2, 3], [2, 3, 4]), ([5, 6, 2], [6, 2, 3])];
    let base = pretrained_base()?;
    let mut tuned = base.clone();
    let before = average_loss(&tuned, &data)?;
    pretrain(&mut tuned, &data, 120, 0.08)?;
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
    let held = [([0, 2, 3], [2, 3, 4]), ([4, 6, 2], [6, 2, 3])];
    let generic = [([0, 1, 2], [1, 2, 3]), ([4, 5, 6], [5, 6, 7])];
    println!(
        "LoRA unseen-prefix full-sequence loss {:.4} -> {:.4}; generic retention {:.4} -> {:.4}",
        average_loss(&base, &held)?,
        average_loss(&adapted, &held)?,
        average_loss(&base, &generic)?,
        average_loss(&adapted, &generic)?
    );
    let mut student = Decoder::new(config(4), 7)?;
    let transfer = [1, 2, 3, 5];
    let learning_rate = 0.12;
    let kl_before = mean_teacher_kl(&tuned, &student, &transfer, 1.0)?;
    for step in 0..160 {
        let token = transfer[step % transfer.len()];
        let teacher_probabilities = softmax(&last_logits(&tuned, token)?, 1.0);
        let gradients = distill_gradient(&student, token, &teacher_probabilities)?;
        student.apply_sgd(&gradients, learning_rate)?;
    }
    let kl_after = mean_teacher_kl(&tuned, &student, &transfer, 1.0)?;
    println!("distillation mean KL at tau=1: {kl_before:.5} -> {kl_after:.5}");
    Ok(())
}

/// Run the completed algorithm; the browser never executes this Rust.
pub fn run(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("unexpected experiment argument".into());
    }
    experiment()
        .and_then(|()| advanced_experiment())
        .map_err(|e| e.to_string())
}

/// Exact temperature-scaled gradient via a linear combination of hard-target backwards.
/// For coefficient c_i=-tau*(p_tau_i-q_tau_i), sum(c)=0 and sum(c_i*(p_1-e_i))=tau*(p_tau-q_tau).
fn temperature_gradient(
    student: &Decoder,
    token: usize,
    teacher_logits: &[f32],
    temperature: f32,
) -> Result<Gradients, Box<dyn Error>> {
    if !temperature.is_finite()
        || temperature <= 0.
        || teacher_logits.len() != student.config().vocab_size
        || teacher_logits.iter().any(|x| !x.is_finite())
    {
        return Err("temperature and teacher logits are invalid".into());
    }
    let student_logits = student.forward(&[token])?;
    let p = softmax(&student_logits, temperature);
    let q = softmax(teacher_logits, temperature);
    let logp = log_softmax(&student_logits, temperature);
    let mut result = Gradients {
        loss: temperature * temperature * q.iter().zip(logp).map(|(q, p)| -q * p).sum::<f32>(),
        tokens: 1,
        values: vec![0.; student.parameter_count()],
    };
    for (target, (&p, &q)) in p.iter().zip(&q).enumerate() {
        let g = student.loss_and_gradient(&[token], &[target])?;
        let coefficient = -temperature * (p - q);
        for (a, b) in result.values.iter_mut().zip(g.values) {
            *a += coefficient * b;
        }
    }
    Ok(result)
}
fn advanced_experiment() -> Result<(), Box<dyn Error>> {
    let base = pretrained_base()?;
    let data = [([1, 2, 3], [2, 3, 4]), ([5, 6, 2], [6, 2, 3])];
    let held = [([0, 2, 3], [2, 3, 4]), ([4, 6, 2], [6, 2, 3])];
    let response_loss =
        |model: &Decoder, rows: &[([usize; 3], [usize; 3])]| -> Result<f32, Box<dyn Error>> {
            let mut sum = 0.;
            for (x, y) in rows {
                sum += masked_gradient(model, x, y, &[false, false, true])?.loss;
            }
            Ok(sum / rows.len() as f32)
        };
    let mut tuned = base.clone();
    let train_before = response_loss(&tuned, &data)?;
    let held_before = response_loss(&tuned, &held)?;
    sft(&mut tuned, &data, 120, 0.08)?;
    println!("response-masked SFT: train {train_before:.4} -> {:.4}; unseen prompt prefixes {held_before:.4} -> {:.4}",response_loss(&tuned,&data)?,response_loss(&tuned,&held)?);
    let generic = [([0, 1, 2], [1, 2, 3]), ([4, 5, 6], [5, 6, 7])];
    println!(
        "masked SFT generic-cycle retention loss {:.4} -> {:.4}",
        average_loss(&base, &generic)?,
        average_loss(&tuned, &generic)?
    );
    let mut student = Decoder::new(config(4), 7)?;
    let transfer = [1, 2, 3, 5];
    let before = mean_teacher_kl(&tuned, &student, &transfer, 2.)?;
    let held_before = mean_teacher_kl(&tuned, &student, &[0, 4], 2.)?;
    for step in 0..160 {
        let t = transfer[step % transfer.len()];
        let g = temperature_gradient(&student, t, &tuned.forward(&[t])?, 2.)?;
        student.apply_sgd(&g, 0.08)?;
    }
    println!("tau=2 distillation KL: transfer {before:.5} -> {:.5}; unseen tokens {held_before:.5} -> {:.5}",mean_teacher_kl(&tuned,&student,&transfer,2.)?,mean_teacher_kl(&tuned,&student,&[0,4],2.)?);
    Ok(())
}
pub fn check() -> Result<(), String> {
    let verify = || -> Result<(), Box<dyn Error>> {
        let base = pretrained_base()?;
        let input = [1, 2, 3];
        let a = masked_gradient(&base, &input, &[2, 3, 4], &[false, false, true])?;
        let b = masked_gradient(&base, &input, &[7, 0, 4], &[false, false, true])?;
        if (a.loss - b.loss).abs() > 2e-5
            || a.values
                .iter()
                .zip(b.values)
                .any(|(a, b)| (a - b).abs() > 2e-5)
        {
            return Err("GOAL_NOT_MET: masked prompt targets changed the response gradient; implement prefix-gradient selection".into());
        }
        let mut l = Lora::new(8, 8, 2)?;
        let frozen = base.parameters().to_vec();
        if l.adapted(&base).parameters() != base.parameters() {
            return Err("GOAL_NOT_MET: adapter must start with zero delta".into());
        }
        let before = l.adapted(&base).loss(&input, &[2, 3, 4])?;
        for _ in 0..120 {
            l.step(&base, &input, &[2, 3, 4], 0.4)?;
        }
        if l.adapted(&base).loss(&input, &[2, 3, 4])? >= before || frozen != base.parameters() {
            return Err("GOAL_NOT_MET: LoRA must train factors while freezing the base".into());
        }
        let hidden = [0.5, -1., 0.25, 0.75, -0.2, 0.1, 0.9, -0.4];
        let direct = l.project(&hidden, &base)?;
        let merged = l.adapted(&base);
        let weights = &merged.parameters()[output_span(&merged)];
        let merged_output: Vec<f32> = (0..8)
            .map(|j| (0..8).map(|i| hidden[i] * weights[i * 8 + j]).sum())
            .collect();
        if direct
            .iter()
            .zip(merged_output)
            .any(|(a, b)| (a - b).abs() > 2e-5 + 2e-5 * b.abs())
        {
            return Err("GOAL_NOT_MET: LoRA direct branch and merged projection disagree".into());
        }
        // Nonzero B opens the A gradient path; freeze-A baseline must fail this check.
        let mut probe = Lora::new(8, 8, 2)?;
        probe.b[0] = 0.2;
        let h = 1e-3;
        let mut plus = probe.clone();
        plus.a[0] += h;
        let mut minus = probe.clone();
        minus.a[0] -= h;
        let numeric = (plus.adapted(&base).loss(&[1, 2], &[2, 3])?
            - minus.adapted(&base).loss(&[1, 2], &[2, 3])?)
            / (2. * h);
        let old = probe.a[0];
        probe.step(&base, &[1, 2], &[2, 3], 0.01)?;
        let analytic = (old - probe.a[0]) / 0.01;
        if numeric.abs() < 1e-4 || (numeric - analytic).abs() > 2e-3 {
            return Err(
                "GOAL_NOT_MET: LoRA must update A as well as B using old factors; finite-difference check failed"
                    .into(),
            );
        }
        let student = Decoder::new(config(4), 7)?;
        let teacher = base.forward(&[1])?;
        let g = temperature_gradient(&student, 1, &teacher, 2.)?;
        let idx = output_span(&student).start;
        let h = 1e-3;
        let mut plus = student.clone();
        plus.parameters_mut()[idx] += h;
        let mut minus = student.clone();
        minus.parameters_mut()[idx] -= h;
        let q = softmax(&teacher, 2.);
        let objective = |m: &Decoder| -> Result<f32, Box<dyn Error>> {
            Ok(4.
                * q.iter()
                    .zip(log_softmax(&m.forward(&[1])?, 2.))
                    .map(|(q, p)| -q * p)
                    .sum::<f32>())
        };
        let numeric = (objective(&plus)? - objective(&minus)?) / (2. * h);
        if (numeric - g.values[idx]).abs() > 1e-3 + 1e-3 * numeric.abs() {
            return Err("GOAL_NOT_MET: temperature gradient fails finite difference".into());
        }
        advanced_experiment()?;
        println!("goal: response mask, frozen trained LoRA and temperature gradient pass");
        Ok(())
    };
    verify().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn soft_target_gradient_and_kl_are_consistent() {
        let student = Decoder::new(config(4), 7).unwrap();
        let teacher_probabilities = [0.4, 0.1, 0.05, 0.15, 0.1, 0.05, 0.1, 0.05];
        let gradients = distill_gradient(&student, 1, &teacher_probabilities).unwrap();
        let parameter = output_span(&student).start;
        let h = 1e-3;
        let mut plus = student.clone();
        let mut minus = student.clone();
        plus.parameters_mut()[parameter] += h;
        minus.parameters_mut()[parameter] -= h;
        let loss = |model: &Decoder| {
            teacher_probabilities
                .iter()
                .zip(log_softmax(&last_logits(model, 1).unwrap(), 1.0))
                .map(|(&p, log_q)| -p * log_q)
                .sum::<f32>()
        };
        let numeric = (loss(&plus) - loss(&minus)) / (2.0 * h);
        assert!((numeric - gradients.values[parameter]).abs() < 3e-4 + 1e-3 * numeric.abs());
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
        pretrain(&mut tuned, &data, 80, 0.08).unwrap();
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
        let learning_rate = 0.12;
        for i in 0..120 {
            let t = xs[i % xs.len()];
            let teacher_probabilities = softmax(&last_logits(&tuned, t).unwrap(), 1.0);
            let gradients = distill_gradient(&student, t, &teacher_probabilities).unwrap();
            student.apply_sgd(&gradients, learning_rate).unwrap();
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
                .loss_and_gradient(&[1, 2], &[2, 3])
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

    #[test]
    fn complete_learning_goals() {
        assert!(super::check().is_ok());
    }
}
