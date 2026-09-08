use super::*;

// Independently evaluate the dense decoder: no production forward/cache, routing,
// capacity dispatch, matmul, LayerNorm or GELU helpers are called here.
fn dense_oracle(model: &Model, tokens: &[usize]) -> Vec<f64> {
    let c = model.config;
    assert_eq!(c.experts, 1);
    let d = c.width;
    let p = &model.parameters;
    let affine = |x: &[f64], w: &Range<usize>, b: &Range<usize>| -> Vec<f64> {
        (0..b.len())
            .map(|o| {
                p[b.start + o]
                    + x.iter()
                        .enumerate()
                        .map(|(i, x)| x * p[w.start + i * b.len() + o])
                        .sum::<f64>()
            })
            .collect()
    };
    let norm = |x: &[f64], range: &Range<usize>| -> Vec<f64> {
        let mean = x.iter().sum::<f64>() / d as f64;
        let variance = x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / d as f64;
        x.iter()
            .enumerate()
            .map(|(j, x)| {
                (x - mean) / (variance + 1e-5).sqrt() * p[range.start + j] + p[range.start + d + j]
            })
            .collect()
    };
    let x: Vec<Vec<f64>> = tokens
        .iter()
        .enumerate()
        .map(|(t, &token)| {
            (0..d)
                .map(|j| {
                    p[model.layout.token.start + token * d + j]
                        + p[model.layout.position.start + t * d + j]
                })
                .collect()
        })
        .collect();
    let qkv: Vec<_> = x
        .iter()
        .map(|row| {
            affine(
                &norm(row, &model.layout.norm1),
                &model.layout.qkv_w,
                &model.layout.qkv_b,
            )
        })
        .collect();
    let expert = &model.layout.experts[0];
    let mut logits = Vec::new();
    for i in 0..tokens.len() {
        let scores: Vec<f64> = (0..=i)
            .map(|j| (0..d).map(|k| qkv[i][k] * qkv[j][d + k]).sum::<f64>() / (d as f64).sqrt())
            .collect();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator = scores.iter().map(|x| (x - maximum).exp()).sum::<f64>();
        let context: Vec<_> = (0..d)
            .map(|k| {
                (0..=i)
                    .map(|j| (scores[j] - maximum).exp() / denominator * qkv[j][2 * d + k])
                    .sum()
            })
            .collect();
        let attention = affine(
            &context,
            &model.layout.attention_w,
            &model.layout.attention_b,
        );
        let residual: Vec<_> = x[i].iter().zip(attention).map(|(a, b)| a + b).collect();
        let pre = affine(
            &norm(&residual, &model.layout.norm2),
            &expert.w1,
            &expert.b1,
        );
        let hidden: Vec<_> = pre
            .iter()
            .map(|&z| {
                z * 0.5
                    * (1.0
                        + ((2.0 / std::f64::consts::PI).sqrt() * (z + 0.044715 * z * z * z)).tanh())
            })
            .collect();
        let dense = affine(&hidden, &expert.w2, &expert.b2);
        let block: Vec<_> = residual.iter().zip(dense).map(|(a, b)| a + b).collect();
        logits.extend(affine(
            &norm(&block, &model.layout.norm_final),
            &model.layout.output_w,
            &model.layout.output_b,
        ));
    }
    logits
}

#[test]
fn one_expert_matches_independent_dense_decoder_and_causality() {
    let mut model = Model::new(Config::tiny(1), 96).unwrap();
    for span in [
        &model.layout.norm1,
        &model.layout.norm2,
        &model.layout.norm_final,
    ] {
        model.parameters[span.start] = 1.3;
        model.parameters[span.start + model.config.width + 1] = 0.2;
    }
    let input = [5, 9, 11, 7];
    let actual = model.forward(&input).unwrap();
    let oracle = dense_oracle(&model, &input);
    for (a, b) in actual.iter().zip(oracle) {
        assert!((a - b).abs() < 1e-12);
    }
    let mut changed = input;
    changed[3] = 17;
    let future = model.forward(&changed).unwrap();
    assert_eq!(&actual[..3 * 128], &future[..3 * 128]);
    assert_ne!(&actual[3 * 128..], &future[3 * 128..]);
    assert_eq!(&actual[..2 * 128], model.forward(&input[..2]).unwrap());
    // Also prove sparse inference remains prefix invariant under a restrictive training cap.
    let sparse = Model::new(
        Config {
            capacity_factor: 0.1,
            ..Config::tiny(3)
        },
        7,
    )
    .unwrap();
    let full = sparse.forward(&input).unwrap();
    assert_eq!(&full[..2 * 128], sparse.forward(&input[..2]).unwrap());
}

#[test]
fn every_parameter_gradient_matches_finite_differences_away_from_ties() {
    for selected in 0..2 {
        let config = Config {
            vocab_size: 8,
            context: 3,
            width: 3,
            ff_width: 4,
            experts: 2,
            capacity_factor: 10.0,
            balance_weight: 0.03,
        };
        let mut model = Model::new(config, 92).unwrap();
        model.parameters[model.layout.router_b.start + selected] += 1.0;
        let input = [1, 3, 5];
        let target = [3, 5, 2];
        let analytic = model.loss_and_gradient(&input, &target).unwrap();
        assert!(analytic.routes.iter().all(|&e| e == selected));
        for index in 0..model.parameters.len() {
            let old = model.parameters[index];
            model.parameters[index] = old + 1e-5;
            let plus = model.loss(&input, &target).unwrap();
            model.parameters[index] = old - 1e-5;
            let minus = model.loss(&input, &target).unwrap();
            model.parameters[index] = old;
            let numerical = (plus - minus) / 2e-5;
            assert!(
                (numerical - analytic.values[index]).abs()
                    <= 1e-6 + 1e-4 * numerical.abs().max(analytic.values[index].abs()),
                "parameter {index}: numerical={numerical}, analytic={}",
                analytic.values[index]
            );
        }
        model.config.balance_weight = 0.0;
        let task_only = model.loss_and_gradient(&input, &target).unwrap();
        assert!(task_only.values[model.layout.router_w.clone()]
            .iter()
            .any(|x| x.abs() > 1e-10));
        let untouched = &model.layout.experts[1 - selected];
        assert!(task_only.values[untouched.w1.clone()]
            .iter()
            .all(|&x| x == 0.0));
    }
}

#[test]
fn dropped_tokens_keep_task_loss_and_shared_decoder_gradients() {
    let mut model = Model::new(
        Config {
            vocab_size: 8,
            context: 4,
            width: 3,
            ff_width: 4,
            experts: 2,
            capacity_factor: 0.5,
            balance_weight: 0.0,
        },
        17,
    )
    .unwrap();
    model.parameters[model.layout.router_w.clone()].fill(0.0);
    model.parameters[model.layout.router_b.start] = 1.0;
    model.parameters[model.layout.router_b.start + 1] = 0.0;
    let input = [1, 3, 5, 7];
    let targets = [3, 5, 7, 2];
    let cache = model.forward_cached(&input, true).unwrap();
    let gradient = model.loss_and_gradient(&input, &targets).unwrap();
    let (all_token_loss, _) = cross_entropy_with_gradient_from_logits(
        &cache.logits,
        &targets,
        input.len(),
        model.config.vocab_size,
    )
    .unwrap();
    assert_eq!(gradient.task_loss, all_token_loss);
    assert_eq!(gradient.accepted, [1, 0]);
    assert_eq!(gradient.dropped, 3);

    let first = model.loss_and_gradient(&input[..1], &targets[..1]).unwrap();
    for range in [
        &model.layout.experts[0].w1,
        &model.layout.experts[0].b1,
        &model.layout.experts[0].w2,
        &model.layout.experts[0].b2,
    ] {
        for index in range.clone() {
            assert!((gradient.values[index] - first.values[index] / 4.0).abs() < 1e-12);
        }
    }
    for range in [
        &model.layout.experts[1].w1,
        &model.layout.experts[1].b1,
        &model.layout.experts[1].w2,
        &model.layout.experts[1].b2,
    ] {
        assert!(gradient.values[range.clone()]
            .iter()
            .all(|&value| value == 0.0));
    }
    for position in 1..input.len() {
        let start = model.layout.position.start + position * model.config.width;
        assert!(gradient.values[start..start + model.config.width]
            .iter()
            .any(|value| value.abs() > 1e-12));
    }
}

#[test]
fn checkpoint_rejects_hostile_headers_and_changed_resume_inputs() {
    let path = std::env::temp_dir().join(format!("ch56-hostile-{}.bin", std::process::id()));
    let mut trainer = Trainer::new(Model::new(Config::tiny(2), 1).unwrap(), 2);
    trainer.train_step(b"rust words", 8, 0.05).unwrap();
    trainer.save(&path).unwrap();
    let good = fs::read(&path).unwrap();
    for mutation in 0..5 {
        let mut bad = good.clone();
        match mutation {
            0 => bad[8..16].copy_from_slice(&u64::MAX.to_le_bytes()),
            1 => {
                bad.pop();
            }
            2 => bad.extend_from_slice(&[0]),
            3 => {
                let n = bad.len();
                bad[n - 8..].copy_from_slice(&f64::NAN.to_le_bytes());
            }
            _ => bad[80..88].fill(0), // forbidden RNG state
        }
        fs::write(&path, bad).unwrap();
        assert!(Trainer::load(&path).is_err(), "mutation {mutation}");
    }
    fs::write(&path, good).unwrap();
    let mut restored = Trainer::load(&path).unwrap();
    fs::remove_file(path).unwrap();
    let before = restored.model.parameters.clone();
    assert!(restored.train_step(b"other text", 8, 0.05).is_err());
    assert!(restored.train_step(b"rust words", 8, 0.06).is_err());
    assert!(restored.train_step(b"rust words", 7, 0.05).is_err());
    assert_eq!(before, restored.model.parameters);
    assert!(Model::new(
        Config {
            context: usize::MAX,
            ..Config::tiny(2)
        },
        1
    )
    .is_err());
}

struct FragmentedStream {
    read: std::io::Cursor<Vec<u8>>,
    written: Vec<u8>,
}
impl Read for FragmentedStream {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let len = bytes.len().min(1);
        self.read.read(&mut bytes[..len])
    }
}
impl Write for FragmentedStream {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.written.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn fragmented_requests_receive_complete_200_or_400_responses() {
    for (request, status) in [
        (
            "GET /generate?prompt=rust&tokens=2 HTTP/1.1\r\nHost: localhost\r\n\r\n",
            "200 OK",
        ),
        (
            "GET /generate?prompt=x&prompt=y HTTP/1.1\r\n\r\n",
            "400 Bad Request",
        ),
        (
            "GET /generate?prompt=x HTTP/1.1\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n",
            "400 Bad Request",
        ),
        (
            "GET /generate?prompt=x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n",
            "400 Bad Request",
        ),
        ("GET /generate?prompt=x HTTP/1.1\r\n", "400 Bad Request"),
    ] {
        let mut trainer = Trainer::new(Model::new(Config::tiny(2), 1).unwrap(), 2);
        let mut stream = FragmentedStream {
            read: std::io::Cursor::new(request.as_bytes().to_vec()),
            written: Vec::new(),
        };
        handle_connection(&mut trainer, &mut stream).unwrap();
        let response = String::from_utf8(stream.written).unwrap();
        assert!(response.starts_with(&format!("HTTP/1.1 {status}\r\n")));
        let (header, body) = response.split_once("\r\n\r\n").unwrap();
        assert!(header.contains(&format!("Content-Length: {}", body.len())));
    }
}

#[test]
fn large_common_logits_and_tiny_temperature_are_stable() {
    let (loss, _) = cross_entropy_with_gradient_from_logits(&[1e300, 1e300], &[0], 1, 2).unwrap();
    assert!((loss - 2f64.ln()).abs() < 1e-12);
    for _ in 0..4 {
        assert_eq!(
            sample(&[1.0, 2.0, 0.0], f64::MIN_POSITIVE, &mut Rng::new(7)),
            1
        );
    }
}

#[test]
fn failed_exclusive_save_keeps_existing_temporary_file() {
    let trainer = Trainer::new(Model::new(Config::tiny(2), 1).unwrap(), 2);
    let path = std::env::temp_dir().join(format!("ch56-exclusive-{}.bin", std::process::id()));
    let temporary = path.with_extension(format!("tmp-{}-0", std::process::id()));
    fs::write(&temporary, b"existing file must remain").unwrap();
    assert!(trainer.save(&path).is_err());
    assert_eq!(fs::read(&temporary).unwrap(), b"existing file must remain");
    fs::remove_file(temporary).unwrap();
    let mut invalid = trainer;
    invalid.step = 2;
    assert!(invalid.save(&path).is_err());
    assert!(!path.exists());
}
