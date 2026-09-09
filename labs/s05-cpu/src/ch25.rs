//! Working baseline: one observable timed call. Implement the complete repeated measurement loop.
use crate::common::{self, Timing};
use std::time::Instant;
pub fn measure(
    run: &mut dyn FnMut() -> Result<(), String>,
    warmups: usize,
    repetitions: usize,
) -> Result<Timing, String> {
    if repetitions == 0 || repetitions > 101 || repetitions.is_multiple_of(2) || warmups > 100 {
        return Err("use 1..101 odd samples and at most 100 warmups".into());
    }
    let start = Instant::now();
    run()?;
    Ok(Timing {
        samples: vec![start.elapsed()],
        warmups: 0,
    })
}
pub fn run(args: &[String]) -> Result<(), String> {
    common::demo(common::dense)?;
    let mut y = [0.; 4];
    let mut timing = measure(
        &mut || {
            common::dense(
                std::hint::black_box(&[1., 2., 3., 4., 5., 6.]),
                std::hint::black_box(&[1., 3., 5., 2., 4., 6.]),
                &[0., 0.],
                std::hint::black_box(&mut y),
                common::Shape { m: 2, k: 3, n: 2 },
            )
        },
        3,
        11,
    )?;
    timing.report()?;
    println!("Baseline single-call dense timer: replace measure with warmups and repeated samples; debug timing is not performance evidence. checksum={}",std::hint::black_box(y).iter().sum::<f32>());
    if args.iter().any(|a| a == "--bench") {
        common::benchmark_with("scalar dense threads=1", common::dense, measure)?;
    }
    Ok(())
}
pub fn check() -> Result<(), String> {
    verify_measure(measure)
}
pub type Measure =
    fn(&mut dyn FnMut() -> Result<(), String>, usize, usize) -> Result<Timing, String>;
pub fn verify_measure(measure: Measure) -> Result<(), String> {
    for (warmups, repetitions) in [(3, 11), (2, 5), (0, 1)] {
        let mut calls = 0;
        let result = measure(
            &mut || {
                calls += 1;
                Ok(())
            },
            warmups,
            repetitions,
        )?;
        if calls != warmups + repetitions
            || result.samples.len() != repetitions
            || result.warmups != warmups
        {
            return Err(format!("GOAL_NOT_MET: requested {warmups} warmups + {repetitions} samples; observed {calls} calls, {} samples, {} warmups. Implement the measurement plan.",result.samples.len(),result.warmups));
        }
    }
    if measure(&mut || Err("probe".into()), 1, 3).is_ok() || measure(&mut || Ok(()), 0, 0).is_ok() {
        return Err(
            "GOAL_NOT_MET: measurement must propagate errors and reject zero samples".into(),
        );
    }
    common::verify(common::dense)
}
#[cfg(test)]
mod tests {
    #[test]
    fn baseline_runs_and_solution_meets_goal() -> Result<(), String> {
        super::run(&[])?;
        crate::solutions::ch25::check()
    }
}
