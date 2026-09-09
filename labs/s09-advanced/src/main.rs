fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            if e.starts_with("GOAL_REVIEW_REQUIRED:") {
                std::process::ExitCode::from(3)
            } else {
                std::process::ExitCode::FAILURE
            }
        }
    }
}
fn run() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err("usage: s09-advanced NN [--check] [--solution] [experiment arguments]".into());
    }
    let chapter = args.remove(0);
    let solution = args.iter().any(|x| x == "--solution");
    let check = args.iter().any(|x| x == "--check");
    args.retain(|x| x != "--solution" && x != "--check");
    if check && !args.is_empty() {
        return Err("--check takes no experiment arguments".into());
    }
    macro_rules! chapter {
        ($learner:path, $sol:path) => {{
            use $learner as learner;
            use $sol as solved;
            if check {
                let result = if solution {
                    solved::check()
                } else {
                    learner::check()
                };
                result?;
                if chapter == "49" && !solution { return Err("GOAL_REVIEW_REQUIRED: numerical scan and attention checks passed; inspect associative_scan for offset-doubling rounds reading the immutable previous round, with ordered composition. See lab.manual_checks and chapter 49.".into()); }
                println!(
                    "chapter {} learning-goal checks passed ({})",
                    chapter,
                    if solution { "solution" } else { "learner" }
                );
                Ok(())
            } else {
                println!(
                    "chapter {}: {}",
                    chapter,
                    if solution {
                        "worked solution"
                    } else {
                        "working learner baseline; --check assesses the full goal"
                    }
                );
                if solution {
                    solved::run(&args)
                } else {
                    learner::run(&args)
                }
            }
        }};
    }
    match chapter.as_str() {
        "47" => chapter!(s09_advanced::ch47, s09_advanced::solutions::ch47),
        "48" => chapter!(s09_advanced::ch48, s09_advanced::solutions::ch48),
        "49" => chapter!(s09_advanced::ch49, s09_advanced::solutions::ch49),
        "50" => chapter!(s09_advanced::ch50, s09_advanced::solutions::ch50),
        "51" => chapter!(s09_advanced::ch51, s09_advanced::solutions::ch51),
        "52" => chapter!(s09_advanced::ch52, s09_advanced::solutions::ch52),
        "53" => chapter!(s09_advanced::ch53, s09_advanced::solutions::ch53),
        "54" => chapter!(s09_advanced::ch54, s09_advanced::solutions::ch54),
        "55" => chapter!(s09_advanced::ch55, s09_advanced::solutions::ch55),
        "56" => chapter!(s09_advanced::ch56, s09_advanced::solutions::ch56),
        _ => Err("chapter must be 47..56".into()),
    }
}
