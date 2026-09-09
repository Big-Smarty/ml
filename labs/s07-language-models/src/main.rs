mod attention;
mod byte_model;
mod capstone;
mod ch33;
mod ch34;
mod ch35;
mod ch36;
mod ch37;
mod ch38;
mod ch39;
mod corpus;
mod decoder;
mod tokenizer;
mod training;
mod solutions {
    pub mod ch33;
    pub mod ch34;
    pub mod ch35;
    pub mod ch36;
    pub mod ch37;
    pub mod ch38;
    pub mod ch39;
}
type LabResult<T = ()> = Result<T, String>;
fn ensure(condition: bool, message: impl Into<String>) -> LabResult {
    if condition {
        Ok(())
    } else {
        let message = message.into();
        Err(if let Some(detail) = message.strip_prefix("goal:") {
            format!("GOAL_NOT_MET:{detail}")
        } else {
            message
        })
    }
}
fn main() {
    if let Err(error) = dispatch() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
fn dispatch() -> LabResult {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err("usage: NN [--check] [--solution]; chapters 33–39".into());
    }
    let chapter = args.remove(0);
    let solution = args.iter().any(|a| a == "--solution");
    let check = args.iter().any(|a| a == "--check");
    args.retain(|a| a != "--solution" && a != "--check");
    if check && !args.is_empty() {
        return Err("--check uses fixed fixtures; omit experiment options".into());
    }
    macro_rules! route {
        ($learner:ident) => {{
            if solution {
                if check {
                    solutions::$learner::check()
                } else {
                    solutions::$learner::run(&args)
                }
            } else if check {
                $learner::check()
            } else {
                $learner::run(&args)
            }
        }};
    }
    match chapter.as_str() {
        "33" => route!(ch33),
        "34" => route!(ch34),
        "35" => route!(ch35),
        "36" => route!(ch36),
        "37" => route!(ch37),
        "38" => route!(ch38),
        "39" => route!(ch39),
        _ => Err("choose a chapter from 33 through 39".into()),
    }
}
fn no_args(args: &[String]) -> LabResult {
    ensure(
        args.is_empty(),
        "this chapter takes only --check and --solution",
    )
}
