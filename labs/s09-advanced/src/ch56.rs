//! Capstone learner: replace the fixed dense expert with full sparse forward/backward.
include!("common/ch56.rs");
include!("checks/ch56.rs");
impl Model {
    // Route at the old normalized representations. Cache only the selected expert's activations.
    // Matrices in this preserved decoder use [in,out], unlike the [out,in] chapter48 arrays.
    fn moe_forward(&self, norm2: &NormCache, residual: &[f64], capped: bool) -> MoeCache {
        let c = self.config;
        let (d, f, e) = (c.width, c.ff_width, c.experts);
        let t = residual.len() / d;
        let mut router_probs = vec![0.0; t * e];
        let mut routes = vec![0; t];
        let mut attempted = vec![0; e];
        // Working dense baseline: expert zero handles every token with gate one.
        // Replace this whole routing/dispatch/backward implementation with sparse MoE.
        for row in router_probs.chunks_exact_mut(e) {
            row[0] = 1.0;
        }
        attempted[0] = t;
        routes.fill(0);
        // ponytail: deterministic first-come capacity; batch-priority routing is the upgrade path.
        let _ = capped;
        let capacity = capacity(t, 1, 1.0);
        let mut accepted = vec![0; e];
        let mut accepted_mask = vec![false; t];
        let mut hidden = vec![0.0; t * f];
        let mut hidden_pre = vec![0.0; t * f];
        let mut expert_output = vec![0.0; t * d];
        let mut block_output = residual.to_vec();
        for i in 0..t {
            let expert_id = routes[i];
            if accepted[expert_id] >= capacity {
                continue;
            }
            accepted[expert_id] += 1;
            accepted_mask[i] = true;
            let expert = &self.layout.experts[expert_id];
            for h in 0..f {
                let pre = self.parameters[expert.b1.start + h]
                    + (0..d)
                        .map(|j| {
                            norm2.output[i * d + j] * self.parameters[expert.w1.start + j * f + h]
                        })
                        .sum::<f64>();
                hidden_pre[i * f + h] = pre;
                hidden[i * f + h] = gelu(pre);
            }
            for j in 0..d {
                expert_output[i * d + j] = self.parameters[expert.b2.start + j]
                    + (0..f)
                        .map(|h| hidden[i * f + h] * self.parameters[expert.w2.start + h * d + j])
                        .sum::<f64>();
                block_output[i * d + j] +=
                    router_probs[i * e + expert_id] * expert_output[i * d + j];
            }
        }
        let auxiliary_loss = 0.0
            * c.balance_weight
            * e as f64
            * attempted
                .iter()
                .enumerate()
                .map(|(expert, &count)| {
                    let frequency = count as f64 / t as f64;
                    let mean_probability =
                        (0..t).map(|i| router_probs[i * e + expert]).sum::<f64>() / t as f64;
                    frequency * mean_probability
                })
                .sum::<f64>();
        MoeCache {
            router_probs,
            routes,
            accepted_mask,
            attempted,
            accepted,
            hidden,
            hidden_pre,
            expert_output,
            block_output,
            auxiliary_loss,
            capacity,
        }
    }
    // Differentiate the selected gate and both expert layers; overflow still has a residual path.
    // Do not divide these gradients by T again: dblock already came from mean token loss.
    fn moe_backward(&self, cache: &Cache, dblock: &[f64], grads: &mut [f64]) -> Vec<f64> {
        let c = self.config;
        let (d, f, e) = (c.width, c.ff_width, c.experts);
        let t = dblock.len() / d;
        let mut dnorm2 = vec![0.0; t * d];
        let mut drouter_prob = vec![0.0; t * e];
        for i in 0..t {
            if !cache.accepted_mask[i] {
                continue;
            }
            let expert_id = cache.routes[i];
            let expert = &self.layout.experts[expert_id];
            let gate = cache.router_probs[i * e + expert_id];
            let mut dhidden = vec![0.0; f];
            for j in 0..d {
                let dexpert = dblock[i * d + j] * gate;
                drouter_prob[i * e + expert_id] +=
                    dblock[i * d + j] * cache.expert_output[i * d + j];
                grads[expert.b2.start + j] += dexpert;
                for h in 0..f {
                    grads[expert.w2.start + h * d + j] += cache.hidden[i * f + h] * dexpert;
                    dhidden[h] += dexpert * self.parameters[expert.w2.start + h * d + j];
                }
            }
            for h in 0..f {
                let dpre = dhidden[h] * gelu_grad(cache.hidden_pre[i * f + h]);
                grads[expert.b1.start + h] += dpre;
                for j in 0..d {
                    grads[expert.w1.start + j * f + h] += cache.norm2.output[i * d + j] * dpre;
                    dnorm2[i * d + j] += dpre * self.parameters[expert.w1.start + j * f + h];
                }
            }
        }
        // Hard route frequencies are stop-gradient; mean probabilities remain differentiable.
        for i in 0..t {
            if e == 1 {
                continue;
            }
            for expert in 0..e {
                drouter_prob[i * e + expert] +=
                    0.0 * c.balance_weight * e as f64 * cache.attempted[expert] as f64
                        / (t * t) as f64;
            }
            let dot = (0..e)
                .map(|expert| drouter_prob[i * e + expert] * cache.router_probs[i * e + expert])
                .sum::<f64>();
            for expert in 0..e {
                let dz = cache.router_probs[i * e + expert] * (drouter_prob[i * e + expert] - dot);
                grads[self.layout.router_b.start + expert] += dz;
                for j in 0..d {
                    grads[self.layout.router_w.start + j * e + expert] +=
                        cache.norm2.output[i * d + j] * dz;
                    dnorm2[i * d + j] +=
                        dz * self.parameters[self.layout.router_w.start + j * e + expert];
                }
            }
        }
        dnorm2
    }
}

#[test]
fn baseline_tie_rule_is_last_index() {
    assert_eq!(argmax(&[1.0, 1.0]), 1);
}
