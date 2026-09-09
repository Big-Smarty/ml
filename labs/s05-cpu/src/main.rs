#![deny(unsafe_op_in_unsafe_fn)]
mod ch25;
mod ch26;
mod ch27;
mod ch28;
mod common;
mod training;
mod solutions {
    pub mod ch25;
    pub mod ch26;
    pub mod ch27;
    pub mod ch28;
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(if error.starts_with("GOAL_REVIEW_REQUIRED:") {
            3
        } else {
            1
        });
    }
}
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let chapter = args
        .first()
        .ok_or("usage: s05-cpu NN [--check] [--solution] [--bench] [--avx512]")?;
    if args[1..]
        .iter()
        .any(|a| !["--check", "--solution", "--bench", "--avx512"].contains(&a.as_str()))
    {
        return Err("unknown argument".into());
    }
    if args.iter().any(|a| a == "--avx512") && chapter != "28" {
        return Err("--avx512 belongs to chapter 28".into());
    }
    let solution = args.iter().any(|a| a == "--solution");
    let check = args.iter().any(|a| a == "--check");
    if check && args.iter().any(|a| a == "--bench" || a == "--avx512") {
        return Err("run --check separately from --bench or --avx512".into());
    }
    let (run, verify): (common::Run, common::Check) = match (chapter.as_str(), solution) {
        ("25", false) => (ch25::run, ch25::check),
        ("26", false) => (ch26::run, ch26::check),
        ("27", false) => (ch27::run, ch27::check),
        ("28", false) => (ch28::run, ch28::check),
        ("25", true) => (solutions::ch25::run, solutions::ch25::check),
        ("26", true) => (solutions::ch26::run, solutions::ch26::check),
        ("27", true) => (solutions::ch27::run, solutions::ch27::check),
        ("28", true) => (solutions::ch28::run, solutions::ch28::check),
        _ => return Err("chapter must be 25, 26, 27 or 28".into()),
    };
    if check {
        verify()?;
        println!("chapter {chapter}: automatic checks passed; inspect any source-review criteria in the lesson");
        Ok(())
    } else {
        run(&args[1..])
    }
}
