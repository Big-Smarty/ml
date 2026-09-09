mod ch20;
mod ch21;
mod ch22;
mod ch23;
mod ch24;
mod recommendation;
mod representation;
mod residual;
mod sequence;
mod vision;
mod solutions {
    pub mod ch20;
    pub mod ch21;
    pub mod ch22;
    pub mod ch23;
    pub mod ch24;
}

fn close(label: &str, actual: f64, expected: f64) -> Result<(), String> {
    let tolerance = 1e-6 + 1e-4 * expected.abs();
    if !actual.is_finite() || !expected.is_finite() || (actual - expected).abs() > tolerance {
        return Err(format!(
            "GOAL_NOT_MET: {label}: got {actual:.9}, expected {expected:.9}, tolerance {tolerance:.2e}"
        ));
    }
    Ok(())
}
fn dispatch() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err("usage: 20|21|22|23|24 [--check] [--solution] [experiment arguments]".into());
    }
    let chapter = args.remove(0);
    let solution = args.iter().any(|s| s == "--solution");
    let check = args.iter().any(|s| s == "--check");
    args.retain(|s| s != "--solution" && s != "--check");
    if check && !args.is_empty() {
        return Err("--check uses fixed independent inputs; omit experiment arguments".into());
    }
    type Run = fn(&[String]) -> Result<(), String>;
    type Check = fn() -> Result<(), String>;
    let (run, verify): (Run, Check) = match (chapter.as_str(), solution) {
        ("20", false) => (ch20::run, ch20::check),
        ("20", true) => (solutions::ch20::run, solutions::ch20::check),
        ("21", false) => (ch21::run, ch21::check),
        ("21", true) => (solutions::ch21::run, solutions::ch21::check),
        ("22", false) => (ch22::run, ch22::check),
        ("22", true) => (solutions::ch22::run, solutions::ch22::check),
        ("23", false) => (ch23::run, ch23::check),
        ("23", true) => (solutions::ch23::run, solutions::ch23::check),
        ("24", false) => (ch24::run, ch24::check),
        ("24", true) => (solutions::ch24::run, solutions::ch24::check),
        _ => return Err(format!("unknown chapter {chapter}; expected20..24")),
    };
    println!(
        "{} {}",
        chapter,
        if solution {
            "completed solution"
        } else {
            "learner implementation (supplied simpler baseline until edited)"
        }
    );
    if check {
        verify()
    } else {
        run(&args)
    }
}
fn main() -> std::process::ExitCode {
    match dispatch() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn supplied_baselines_run_without_goal_completion() {
        for run in [
            super::ch20::run,
            super::ch21::run,
            super::ch22::run,
            super::ch23::run,
            super::ch24::run,
        ] {
            assert!(run(&[]).is_ok());
        }
    }
    #[test]
    fn completed_goals_pass() {
        for check in [
            super::solutions::ch20::check,
            super::solutions::ch21::check,
            super::solutions::ch22::check,
            super::solutions::ch23::check,
            super::solutions::ch24::check,
        ] {
            assert!(check().is_ok());
        }
    }
    #[test]
    fn horizon_boundaries_are_checked_before_training() {
        for value in ["0", "41", "nan", "1.5"] {
            assert!(super::ch23::run(&["--horizon".into(), value.into()]).is_err());
        }
        assert!(super::solutions::ch23::run(&["--horizon".into(), "1".into()]).is_ok());
    }
    #[test]
    fn rejects_unknown_experiment_arguments() {
        for run in [
            super::ch20::run,
            super::ch21::run,
            super::ch22::run,
            super::ch23::run,
            super::ch24::run,
        ] {
            assert!(run(&["--unknown".into()]).is_err());
        }
    }
}
