//! Worked solution: Cache selected full-softmax gates and admitted expert activations during the complete decoder forward. Backward combines residual, both expert layers, task gate, auxiliary balance, and router-input derivatives before returning to normalization. Source and finite-difference review cover the complete model; inference disables capacity drops to preserve causal prefixes.
//! Compare intermediate quantities with the lesson before using the final goal check.
//! Worked sparse decoder solution: selected full-softmax gate, capped training, full gradients.
include!("../common/ch56.rs");
include!("../checks/ch56.rs");
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
        for i in 0..t {
            let row = &mut router_probs[i * e..(i + 1) * e];
            if e == 1 {
                row[0] = 1.0;
                attempted[0] += 1;
                continue;
            }
            for (expert, probability) in row.iter_mut().enumerate() {
                *probability = self.parameters[self.layout.router_b.start + expert]
                    + (0..d)
                        .map(|j| {
                            norm2.output[i * d + j]
                                * self.parameters[self.layout.router_w.start + j * e + expert]
                        })
                        .sum::<f64>();
            }
            softmax_in_place(row);
            routes[i] = argmax(row);
            attempted[routes[i]] += 1;
        }
        // ponytail: deterministic first-come capacity; batch-priority routing is the upgrade path.
        let capacity = if capped {
            capacity(t, e, c.capacity_factor)
        } else {
            t
        };
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
        let auxiliary_loss = c.balance_weight
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
                    c.balance_weight * e as f64 * cache.attempted[expert] as f64 / (t * t) as f64;
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
#[cfg(test)]
mod tests {
    use super::*;

    const DATA: &[u8] = b"rust routes tokens. rust learns words. ";

    fn span(model: &Model, name: &str) -> Range<usize> {
        model
            .parameter_spans()
            .into_iter()
            .find(|(candidate, _)| candidate == name)
            .unwrap()
            .1
    }

    #[test]
    fn one_expert_is_dense_feed_forward() {
        let model = Model::new(Config::tiny(1), 7).unwrap();
        let stats = model.routing_stats(&[1, 2, 3, 4]).unwrap();
        assert_eq!(stats.attempted, [4]);
        assert_eq!(stats.accepted, [4]);
        assert_eq!(stats.dropped, 0);
        assert_eq!(stats.mean_selected_gate, 1.0);
        assert_eq!(
            model.loss(&[1, 2], &[2, 3]).unwrap(),
            model.loss_and_gradient(&[1, 2], &[2, 3]).unwrap().task_loss
        );
    }

    #[test]
    fn router_and_expert_gradients_match_finite_differences() {
        let mut model = Model::new(
            Config {
                capacity_factor: 10.0,
                ..Config::tiny(3)
            },
            19,
        )
        .unwrap();
        let input = [7, 11, 13];
        let targets = [11, 13, 17];
        // Push expert 0 away from an argmax tie so the local finite difference keeps routes fixed.
        let router_bias = span(&model, "router_bias");
        model.parameters[router_bias.start] += 0.7;
        let analytic = model.loss_and_gradient(&input, &targets).unwrap();
        let expert = analytic.routes[0];
        let checks = [
            router_bias.start,
            span(&model, &format!("expert.{expert}.ff1_weight")).start,
        ];
        for index in checks {
            let original = model.parameters[index];
            let epsilon = 1e-5;
            model.parameters[index] = original + epsilon;
            let plus = model.loss(&input, &targets).unwrap();
            model.parameters[index] = original - epsilon;
            let minus = model.loss(&input, &targets).unwrap();
            model.parameters[index] = original;
            let numerical = (plus - minus) / (2.0 * epsilon);
            let tolerance = 1e-6 + 1e-4 * numerical.abs().max(analytic.values[index].abs());
            assert!(
                (numerical - analytic.values[index]).abs() <= tolerance,
                "index {index}: numerical {numerical}, analytic {}",
                analytic.values[index]
            );
        }
    }

    #[test]
    fn capacity_counts_attempts_and_drops() {
        let mut model = Model::new(
            Config {
                capacity_factor: 0.5,
                ..Config::tiny(2)
            },
            3,
        )
        .unwrap();
        let router_weight = span(&model, "router_weight");
        model.parameters[router_weight].fill(0.0);
        let router_bias = span(&model, "router_bias");
        model.parameters[router_bias.start] = 1.0;
        model.parameters[router_bias.start + 1] = 0.0;
        let stats = model.routing_stats(&[1, 2, 3, 4]).unwrap();
        assert_eq!(stats.capacity, 1);
        assert_eq!(stats.attempted, [4, 0]);
        assert_eq!(stats.accepted, [1, 0]);
        assert_eq!(stats.dropped, 3);
    }

    #[test]
    fn capacity_rounds_after_division() {
        assert_eq!(capacity(7, 3, 1.25), 3);
        assert_eq!(capacity(8, 3, 1.25), 4);
        assert_eq!(capacity(1, 8, 0.5), 1);
    }

    #[test]
    fn parameter_span_abi_is_stable() {
        let model = Model::new(
            Config {
                vocab_size: 3,
                context: 2,
                width: 2,
                ff_width: 3,
                experts: 2,
                capacity_factor: 1.25,
                balance_weight: 0.01,
            },
            1,
        )
        .unwrap();
        let expected = [
            ("norm1", 10, 14),
            ("norm2", 14, 18),
            ("norm_final", 18, 22),
            ("token_embedding", 0, 6),
            ("position_embedding", 6, 10),
            ("qkv_weight", 22, 34),
            ("qkv_bias", 34, 40),
            ("attention_output_weight", 40, 44),
            ("attention_output_bias", 44, 46),
            ("router_weight", 46, 50),
            ("router_bias", 50, 52),
            ("expert.0.ff1_weight", 52, 58),
            ("expert.0.ff1_bias", 58, 61),
            ("expert.0.ff2_weight", 61, 67),
            ("expert.0.ff2_bias", 67, 69),
            ("expert.1.ff1_weight", 69, 75),
            ("expert.1.ff1_bias", 75, 78),
            ("expert.1.ff2_weight", 78, 84),
            ("expert.1.ff2_bias", 84, 86),
            ("output_weight", 86, 92),
            ("output_bias", 92, 95),
        ];
        let spans = model.parameter_spans();
        assert_eq!(spans.len(), expected.len());
        for ((name, range), &(expected_name, start, end)) in spans.iter().zip(&expected) {
            assert_eq!(
                (name.as_str(), range.start, range.end),
                (expected_name, start, end)
            );
        }
        assert_eq!(model.total_parameters(), 95);
        assert_eq!(model.active_parameters_per_token(), 78);
        let tiny = Model::new(Config::tiny(3), 1).unwrap();
        assert_eq!(tiny.total_parameters(), 3_303);
        assert_eq!(tiny.active_parameters_per_token(), 2_879);
    }

    #[test]
    fn checkpoint_resume_is_exact() {
        let mut original = Trainer::new(Model::new(Config::tiny(3), 8).unwrap(), 9);
        for _ in 0..3 {
            original.train_step(DATA, 8, 0.05).unwrap();
        }
        let path = std::env::temp_dir().join(format!("ch56-resume-{}.bin", std::process::id()));
        original.save(&path).unwrap();
        let mut restored = Trainer::load(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(original.model.parameters, restored.model.parameters);
        assert_eq!(
            (original.step, original.cursor, original.rng),
            (restored.step, restored.cursor, restored.rng)
        );
        original.train_step(DATA, 8, 0.05).unwrap();
        restored.train_step(DATA, 8, 0.05).unwrap();
        assert_eq!(original.model.parameters, restored.model.parameters);
    }

    #[test]
    fn training_changes_trunk_router_and_expert() {
        let mut trainer = Trainer::new(
            Model::new(
                Config {
                    capacity_factor: 10.0,
                    ..Config::tiny(3)
                },
                5,
            )
            .unwrap(),
            6,
        );
        let before = trainer.model.parameters.clone();
        let gradients = trainer.train_step(DATA, 12, 0.05).unwrap();
        for name in [
            "token_embedding",
            "position_embedding",
            "qkv_weight",
            "attention_output_weight",
            "norm1",
            "norm2",
            "norm_final",
            "output_weight",
            "router_weight",
        ] {
            let range = span(&trainer.model, name);
            assert_ne!(&before[range.clone()], &trainer.model.parameters[range]);
        }
        let selected = gradients.routes[0];
        let range = span(&trainer.model, &format!("expert.{selected}.ff1_weight"));
        assert_ne!(&before[range.clone()], &trainer.model.parameters[range]);
    }

    #[test]
    fn request_boundary_is_strict() {
        assert_eq!(
            parse_request("GET /generate?prompt=rust+moe&tokens=4 HTTP/1.1\r\n\r\n").unwrap(),
            ("rust moe".into(), 4)
        );
        assert!(parse_request("POST /generate?prompt=x HTTP/1.1\r\n\r\n").is_err());
        assert!(parse_request("GET /generate?prompt=x&tokens=999 HTTP/1.1\r\n\r\n").is_err());
    }
}

#[cfg(test)]
#[path = "../common/verification56.rs"]
mod verification;
