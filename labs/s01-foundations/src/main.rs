mod args;
mod ch01;
mod ch02;
mod ch03;
mod ch04;
mod ch05;
mod ch06;
mod classifier;
mod scalar;
mod vector;
mod solutions {
    pub mod ch01;
    pub mod ch02;
    pub mod ch03;
    pub mod ch04;
    pub mod ch05;
    pub mod ch06;
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
fn dispatch() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let chapter = args
        .first()
        .ok_or("usage: cargo run -- 01 [--solution] [--check]")?;
    let solution = args.iter().any(|s| s == "--solution");
    let check = args.iter().any(|s| s == "--check");
    let experiment: Vec<String> = args[1..]
        .iter()
        .filter(|s| !["--solution", "--check"].contains(&s.as_str()))
        .cloned()
        .collect();
    if check && !experiment.is_empty() {
        return Err(
            "--check uses fixed inspectable fixtures; run experiments without --check".into(),
        );
    }
    match (chapter.as_str(), solution, check) {
        ("01", false, false) => ch01::run(&experiment),
        ("01", false, true) => ch01::check(),
        ("01", true, false) => solutions::ch01::run(&experiment),
        ("01", true, true) => solutions::ch01::check(),
        ("02", false, false) => ch02::run(&experiment),
        ("02", false, true) => ch02::check(),
        ("02", true, false) => solutions::ch02::run(&experiment),
        ("02", true, true) => solutions::ch02::check(),
        ("03", false, false) => ch03::run(&experiment),
        ("03", false, true) => ch03::check(),
        ("03", true, false) => solutions::ch03::run(&experiment),
        ("03", true, true) => solutions::ch03::check(),
        ("04", false, false) => ch04::run(&experiment),
        ("04", false, true) => ch04::check(),
        ("04", true, false) => solutions::ch04::run(&experiment),
        ("04", true, true) => solutions::ch04::check(),
        ("05", false, false) => ch05::run(&experiment),
        ("05", false, true) => ch05::check(),
        ("05", true, false) => solutions::ch05::run(&experiment),
        ("05", true, true) => solutions::ch05::check(),
        ("06", false, false) => ch06::run(&experiment),
        ("06", false, true) => ch06::check(),
        ("06", true, false) => solutions::ch06::run(&experiment),
        ("06", true, true) => solutions::ch06::check(),
        _ => Err(format!("unknown chapter {chapter}; choose01–06")),
    }
}
