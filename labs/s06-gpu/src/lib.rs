//! Four CPU-first experiments with explicitly requested hardware execution.
pub mod ch29;
pub mod ch30;
pub mod ch31;
pub mod ch32;
pub mod host;
pub mod model;
pub mod solutions;
use host::{Host, Kernels, Plan};
use model::{INITIAL, TRAIN_DATA};
/// Named experiment size: change to 257 for the chapter32 transfer experiment.
pub const PERFORMANCE_ELEMENT_COUNT: usize = 65_537;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn validate_pair(a: &[f32], b: &[f32]) -> Result<()> {
    if a.is_empty() || a.len() != b.len() || a.iter().chain(b).any(|x| !x.is_finite()) {
        return Err("vectors must have equal nonzero lengths and finite values".into());
    }
    Ok(())
}

pub fn close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) -> Result<f32> {
    if actual.len() != expected.len() {
        return Err("output length differs from oracle".into());
    }
    let mut max_error = 0.0_f32;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || !e.is_finite() || (a - e).abs() > atol + rtol * e.abs() {
            return Err(format!("index {i}: actual={a}, expected={e}, tolerance={} ; inspect learner arithmetic/indices", atol + rtol * e.abs()).into());
        }
        max_error = max_error.max((a - e).abs());
    }
    Ok(max_error)
}

pub fn kernels(solution: bool) -> Kernels {
    if solution {
        Kernels {
            vector: solutions::ch29::SHADER,
            matmul: solutions::ch30::MATMUL,
            reduce: solutions::ch30::REDUCE,
            training: solutions::ch31::SHADER,
            fused: solutions::ch32::SHADER,
        }
    } else {
        Kernels {
            vector: ch29::SHADER,
            matmul: ch30::MATMUL,
            reduce: ch30::REDUCE,
            training: ch31::SHADER,
            fused: ch32::SHADER,
        }
    }
}

/// Compare f32 analytic gradients to independent f64 central differences.
fn gradient_check(
    gradient: fn(
        &model::Model,
        &[model::Example],
    ) -> std::result::Result<model::Gradient, &'static str>,
) -> Result<()> {
    fn loss(p: &[f64; 17]) -> f64 {
        TRAIN_DATA
            .iter()
            .map(|(x, y)| {
                let z = p[16]
                    + (0..4)
                        .map(|j| {
                            p[12 + j]
                                * (p[2 * j] * f64::from(x[0])
                                    + p[2 * j + 1] * f64::from(x[1])
                                    + p[8 + j])
                                    .tanh()
                        })
                        .sum::<f64>();
                z.max(0.0) - z * f64::from(*y) + (-z.abs()).exp().ln_1p()
            })
            .sum::<f64>()
            / TRAIN_DATA.len() as f64
    }
    let analytic = gradient(&INITIAL, &TRAIN_DATA)?;
    let mut max_error = 0.0_f64;
    for i in 0..17 {
        let mut plus = INITIAL.parameters.map(f64::from);
        let mut minus = plus;
        plus[i] += 1e-5;
        minus[i] -= 1e-5;
        let numeric = (loss(&plus) - loss(&minus)) / 2e-5;
        let error = (f64::from(analytic.parameters[i]) - numeric).abs();
        if error > 1e-6 + 1e-4 * numeric.abs() {
            return Err(format!("GOAL_NOT_MET: gradient[{i}]={} but central difference={numeric:.7}; implement hidden/input chain rule in ch31.rs", analytic.parameters[i]).into());
        }
        max_error = max_error.max(error);
    }
    println!("17 gradient probes: max error={max_error:.3e} against f64 central differences");
    Ok(())
}

pub fn run(chapter: &str, solution: bool, gpu_requested: bool, check: bool) -> Result<()> {
    println!(
        "chapter {chapter}: {} / {}",
        if solution {
            "solution"
        } else {
            "learner baseline"
        },
        if gpu_requested {
            "real hardware requested"
        } else {
            "CPU only; hardware not verified"
        }
    );
    let gpu = if gpu_requested {
        Some(Host::new(kernels(solution))?)
    } else {
        None
    };
    // Only numerical goal comparisons receive the goal marker; runtime errors propagate unchanged.
    let compare = |actual: &[f32], expected: &[f32], atol, rtol| {
        close(actual, expected, atol, rtol).map_err(|error| {
            if check {
                format!("GOAL_NOT_MET: {error}").into()
            } else {
                error
            }
        })
    };
    match chapter {
        "29" => {
            let compute = if solution {
                solutions::ch29::compute
            } else {
                ch29::compute
            };
            let a = [1.0, -2.0, 3.5, 8.0, 0.25];
            let b = [2.0, 5.0, -0.5, 1.0, 0.75];
            let cpu = compute(&a, &b, 0.5)?;
            println!("left={a:?}, right={b:?}, scale=0.5 -> {cpu:?}");
            if let Some(gpu) = &gpu {
                let actual = gpu.vector(&a, &b, 0.5)?;
                let error = compare(&actual, &cpu, 1e-6, 1e-6)?;
                println!("hardware readback={actual:?}, max CPU difference={error:.3e}");
            }
            if check {
                for n in [1, 64, 67, 129] {
                    let a: Vec<_> = (0..n).map(|i| i as f32 * 0.25 - 3.0).collect();
                    let b: Vec<_> = (0..n).map(|i| 5.0 - i as f32 * 0.125).collect();
                    for scale in [0.5, -2.0] {
                        let expected: Vec<_> =
                            a.iter().zip(&b).map(|(a, b)| scale * a + b).collect();
                        compare(&compute(&a, &b, scale)?, &expected, 1e-6, 1e-6)?;
                        if let Some(gpu) = &gpu {
                            println!(
                                "N={n}, scale={scale}, hardware max error={}",
                                compare(&gpu.vector(&a, &b, scale)?, &expected, 1e-6, 1e-6)?
                            );
                        }
                    }
                }
            }
        }
        "30" => {
            let matmul = if solution {
                solutions::ch30::matmul
            } else {
                ch30::matmul
            };
            let reduce = if solution {
                solutions::ch30::reduce
            } else {
                ch30::reduce
            };
            println!(
                "A=[[1,2,3],[4,5,6]], B=[[7,8],[9,10],[11,12]], C={:?}",
                matmul(
                    &[1., 2., 3., 4., 5., 6.],
                    &[7., 8., 9., 10., 11., 12.],
                    2,
                    3,
                    2
                )?
            );
            println!(
                "sum([3,1,4,1,5,9,2,6])={}",
                reduce(&[3., 1., 4., 1., 5., 9., 2., 6.])?
            );
            if check || gpu.is_some() {
                for (m, k, n) in [(3, 17, 5), (17, 19, 33)] {
                    let a: Vec<_> = (0..m * k).map(|i| (i % 13) as f32 / 7.0 - 0.8).collect();
                    let b: Vec<_> = (0..k * n).map(|i| (i % 11) as f32 / 9.0 - 0.6).collect();
                    let expected = ::ch30::matmul_scalar(&a, &b, m, k, n)?;
                    compare(&matmul(&a, &b, m, k, n)?, &expected, 1e-5, 1e-4)?;
                    if let Some(gpu) = &gpu {
                        println!(
                            "{m}x{k} x {k}x{n}: hardware max error={:.3e}",
                            compare(&gpu.matmul(&a, &b, m, k, n)?, &expected, 1e-5, 1e-4)?
                        );
                    }
                }
                for n in [1, 600, 777] {
                    let x: Vec<_> = (0..n).map(|i| i as f32 * 0.125 - 1.0).collect();
                    let expected = ::ch30::reduce_sum_scalar(&x)?;
                    compare(&[reduce(&x)?], &[expected], 1e-4, 1e-5)?;
                    if let Some(gpu) = &gpu {
                        println!(
                            "reduction N={n}: hardware max error={}",
                            compare(&[gpu.reduce_sum(&x)?], &[expected], 1e-4, 1e-5)?
                        );
                    }
                }
                println!("Numerical parity only. Inspect cooperative loads, two uniform tile barriers, tree reduction and no early edge return to establish the implementation goal.");
            }
            if check && !solution {
                return Err("GOAL_REVIEW_REQUIRED: numeric parity passed; review ch30.rs for shared tile loads, a halving reduction tree, uniform barriers, edge-lane participation and distinct output ownership. Attach the 600→2→1 and 17x19x33 traces.".into());
            }
        }
        "31" => {
            let train = if solution {
                solutions::ch31::train
            } else {
                ch31::train
            };
            let trained = train(INITIAL.clone(), &TRAIN_DATA, 800, 0.3)?;
            println!(
                "2-4-1 tanh, 8 examples, 800 full-batch steps, rate=0.3: BCE {:.6} -> {:.6}",
                INITIAL.loss(&TRAIN_DATA)?,
                trained.loss(&TRAIN_DATA)?
            );
            let transfer = [
                ([-0.6, -0.9], 0.0),
                ([-0.9, 0.7], 1.0),
                ([0.7, -0.9], 1.0),
                ([0.9, 0.6], 0.0),
            ];
            for &(x, y) in &transfer {
                println!("unseen {x:?}: p={:.4}, target={y}", trained.probability(x));
            }
            if let Some(gpu) = &gpu {
                for (data, steps) in [
                    (&TRAIN_DATA[..], 1),
                    (&TRAIN_DATA[..], 80),
                    (&TRAIN_DATA[1..4], 3),
                    (&TRAIN_DATA[..], 800),
                ] {
                    let actual = gpu.train(data, steps, 0.3)?;
                    let expected = train(INITIAL.clone(), data, steps, 0.3)?;
                    let error =
                        compare(&actual.model.parameters, &expected.parameters, 2e-4, 1e-4)?;
                    compare(&[actual.losses[steps]], &[expected.loss(data)?], 2e-4, 1e-4)?;
                    println!("resident N={}, steps={steps}: BCE {:.6} -> {:.6}, max parameter error={error:.3e}",data.len(),actual.losses[0],actual.losses[steps]);
                }
            }
            if check {
                gradient_check(if solution {
                    solutions::ch31::gradient
                } else {
                    ch31::gradient
                })?;
                if trained.loss(&TRAIN_DATA)? >= INITIAL.loss(&TRAIN_DATA)? * 0.3 {
                    return Err("GOAL_NOT_MET: goal: reduce mean BCE by at least 70%; fixed hidden features are the baseline".into());
                }
                for &(x, y) in &transfer {
                    if (trained.probability(x) >= 0.5) != (y == 1.0) {
                        return Err(format!(
                            "GOAL_NOT_MET: transfer classification failed at {x:?}"
                        )
                        .into());
                    }
                }
            }
        }
        "32" => {
            let schedule = if solution {
                solutions::ch32::schedule()
            } else {
                ch32::schedule()
            };
            let input: Vec<_> = (0..PERFORMANCE_ELEMENT_COUNT)
                .map(|i| (i % 257) as f32 / 32.0 - 4.0)
                .collect();
            let expected = host::affine_relu_scalar(&input);
            println!(
                "affine + ReLU, N={}, small fixture [-1,0,2] -> {:?}; planned runs={}",
                input.len(),
                host::affine_relu_scalar(&[-1.0, 0.0, 2.0]),
                schedule.len()
            );
            let mut separate = Vec::new();
            let mut fused = Vec::new();
            let mut mixed = Vec::new();
            for &(plan, warmup) in &schedule {
                if let Some(gpu) = &gpu {
                    let end_to_end = std::time::Instant::now();
                    let measurement = gpu.affine_relu_with_gpu(&input, plan)?;
                    let total = end_to_end.elapsed();
                    let max_error = compare(
                        &measurement.output,
                        &expected,
                        if measurement.used_f16 { 0.01 } else { 1e-5 },
                        1e-6,
                    )?;
                    std::hint::black_box(&measurement.output);
                    if !warmup {
                        println!("{plan:?}: call including allocations/upload/readback/timestamp decode={total:?}, max error={max_error:.3e}, f16={}",measurement.used_f16);
                        match plan {
                            Plan::Separate => separate.push(measurement),
                            Plan::FusedF32 => fused.push(measurement),
                            Plan::Mixed => mixed.push(measurement),
                        }
                    }
                }
            }
            for (label, samples) in [("separate", separate), ("fused", fused), ("mixed", mixed)] {
                if !samples.is_empty() {
                    summary(label, &samples);
                }
            }
            if check {
                let mut measuring = false;
                for &(_, warmup) in &schedule {
                    if warmup && measuring {
                        return Err("GOAL_NOT_MET: warmups must precede measured runs".into());
                    }
                    measuring |= !warmup;
                }
                for plan in [Plan::Separate, Plan::FusedF32, Plan::Mixed] {
                    let warmups = schedule.iter().filter(|&&(p, w)| p == plan && w).count();
                    let samples = schedule.iter().filter(|&&(p, w)| p == plan && !w).count();
                    if warmups < 3 || samples < 7 {
                        return Err(format!("GOAL_NOT_MET: {plan:?}: {warmups} warmups/{samples} samples; design at least 3/7 per plan and implement the fused WGSL before selecting it").into());
                    }
                }
                let measured: Vec<_> = schedule
                    .iter()
                    .filter(|(_, w)| !*w)
                    .map(|(p, _)| p)
                    .collect();
                if measured.windows(2).any(|p| p[0] == p[1]) {
                    return Err(
                        "GOAL_NOT_MET: alternate measured plans to reduce order confounding".into(),
                    );
                }
                if let Some(gpu) = &gpu {
                    let non_binary: Vec<_> = (0..257).map(|i| i as f32 / 29.0 - 4.0).collect();
                    for plan in [Plan::Separate, Plan::FusedF32, Plan::Mixed] {
                        let result = gpu.affine_relu_with_gpu(&non_binary, plan)?;
                        let error = compare(
                            &result.output,
                            &host::affine_relu_scalar(&non_binary),
                            if result.used_f16 { 0.01 } else { 1e-5 },
                            1e-6,
                        )?;
                        println!(
                            "non-binary fixture N=257 {plan:?}: f16={}, max error={error:.3e}",
                            result.used_f16
                        );
                    }
                    let boundary = [-1.0, -1.0 / 6.0, 0.0, 2.0, 50_000.0];
                    for plan in [Plan::Separate, Plan::FusedF32, Plan::Mixed] {
                        let result = gpu.affine_relu_with_gpu(&boundary, plan)?;
                        compare(
                            &result.output,
                            &host::affine_relu_scalar(&boundary),
                            1e-5,
                            1e-6,
                        )?;
                        if result.used_f16 {
                            return Err("GOAL_NOT_MET: unsafe half range selected".into());
                        }
                    }
                }
                println!("Schedule evidence passed; speed and shader correctness require --gpu. No speed threshold is a learning goal.");
            }
        }
        _ => return Err("chapter must be 29, 30, 31, or 32".into()),
    }
    if check && !solution && !gpu_requested {
        return Err(format!("GOAL_REVIEW_REQUIRED: chapter {chapter} CPU criteria passed; inspect the actual learner WGSL against lab.manual_checks and run --check --gpu for hardware parity. CPU calculations do not execute a shader.").into());
    }
    if check {
        println!("requested automatic checks passed; consult lab.manual_checks for source/trace criteria. CPU-only runs do not verify GPU execution");
    }
    Ok(())
}

fn summary(label: &str, samples: &[host::Measurement]) {
    let mut wall: Vec<_> = samples.iter().map(|x| x.cpu_wall_time).collect();
    wall.sort_unstable();
    println!(
        "{label}: encoding-through-output-map wall median={:?}, range={:?}..{:?}",
        wall[wall.len() / 2],
        wall[0],
        wall[wall.len() - 1]
    );
    let times: Option<Vec<_>> = samples.iter().map(|x| x.device_time_nanoseconds).collect();
    if let Some(mut times) = times {
        times.sort_by(f64::total_cmp);
        println!(
            "{label}: summed device pass ns median={:.0}, range={:.0}..{:.0}",
            times[times.len() / 2],
            times[0],
            times[times.len() - 1]
        );
    } else {
        println!("{label}: device timestamps unsupported");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baselines_and_completed_cpu_goals() -> Result<()> {
        for ch in ["29", "30", "31", "32"] {
            run(ch, false, false, false)?;
            run(ch, true, false, true)?;
        }
        assert!(run("29", false, false, true).is_err());
        assert!(run("31", false, false, true).is_err());
        assert!(run("32", false, false, true).is_err());
        Ok(())
    }
    #[test]
    fn boundaries_and_known_values() -> Result<()> {
        assert!(ch29::compute(&[], &[], 1.0).is_err());
        assert!(ch29::compute(&[f32::NAN], &[1.0], 1.0).is_err());
        assert!(ch29::compute(&[1.0], &[1.0, 2.0], 1.0).is_err());
        assert!(ch30::matmul(&[], &[], 0, 3, 2).is_err());
        assert!(ch30::matmul(&[], &[], usize::MAX, 3, 2).is_err());
        assert!(ch30::reduce(&[]).is_err());
        assert!(ch31::train(INITIAL.clone(), &[], 3, 0.3).is_err());
        assert!(ch31::train(INITIAL.clone(), &TRAIN_DATA, 1, f32::NAN).is_err());
        assert_eq!(model::binary_cross_entropy_from_logit(1000.0, 0.0)?, 1000.0);
        assert_eq!(
            model::binary_cross_entropy_from_logit(-1000.0, 1.0)?,
            1000.0
        );
        assert_eq!(
            host::affine_relu_scalar(&[-1.0, -1.0 / 6.0, 0.0, 2.0]),
            [0.0, 0.0, 0.25, 3.25]
        );
        assert!(!host::f16_arithmetic_is_safe(&[50_000.0]));
        assert!(close(&[f32::NAN], &[1.0], 1e-5, 1e-5).is_err());
        Ok(())
    }
    #[test]
    #[ignore = "requires a hardware Vulkan adapter; unsupported is an error, not a pass"]
    fn real_hardware_agreement_and_training() -> Result<()> {
        for ch in ["29", "30", "31", "32"] {
            run(ch, true, true, true)?;
        }
        Ok(())
    }
}
