//! Warmups execute the same call but are not samples. Reserve sample storage before measuring.
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
    for _ in 0..warmups {
        run()?;
    }
    let mut samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let start = Instant::now();
        run()?;
        let elapsed = start.elapsed();
        samples.push(elapsed);
    }
    Ok(Timing { samples, warmups })
}
pub fn run(args: &[String]) -> Result<(), String> {
    common::demo(common::dense)?;
    if args.iter().any(|a| a == "--bench") {
        common::benchmark("scalar dense threads=1", common::dense)?;
    }
    Ok(())
}
pub fn check() -> Result<(), String> {
    crate::ch25::verify_measure(measure)
}
