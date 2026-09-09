mod ch07;
mod ch08;
mod ch09;
mod ch10;
mod ch11;
mod data;
mod linear;
mod mlp;
mod tape;
mod tensor;
mod xor;
mod solutions {
    pub mod ch07;
    pub mod ch08;
    pub mod ch09;
    pub mod ch10;
    pub mod ch11;
}
pub fn close(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= 1e-6 + 1e-4 * expected.abs()
}
fn main() -> std::process::ExitCode {
    match dispatch() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}
fn dispatch() -> Result<(), String> {
    let all: Vec<String> = std::env::args().skip(1).collect();
    let chapter = all
        .first()
        .ok_or("usage: cargo run -- NN [--solution] [--check]")?;
    let solution = all[1..].iter().any(|s| s == "--solution");
    let check = all[1..].iter().any(|s| s == "--check");
    let args: Vec<String> = all[1..]
        .iter()
        .filter(|s| s.as_str() != "--solution" && s.as_str() != "--check")
        .cloned()
        .collect();
    if check && !args.is_empty() {
        return Err("this command accepts only --solution and --check".into());
    }
    match (chapter.as_str(), solution, check) {
        ("07", false, false) => ch07::run(&args),
        ("07", false, true) => ch07::check(),
        ("07", true, false) => solutions::ch07::run(&args),
        ("07", true, true) => solutions::ch07::check(),
        ("08", false, false) => ch08::run(&args),
        ("08", false, true) => ch08::check(),
        ("08", true, false) => solutions::ch08::run(&args),
        ("08", true, true) => solutions::ch08::check(),
        ("09", false, false) => ch09::run(&args),
        ("09", false, true) => ch09::check(),
        ("09", true, false) => solutions::ch09::run(&args),
        ("09", true, true) => solutions::ch09::check(),
        ("10", false, false) => ch10::run(&args),
        ("10", false, true) => ch10::check(),
        ("10", true, false) => solutions::ch10::run(&args),
        ("10", true, true) => solutions::ch10::check(),
        ("11", false, false) => ch11::run(&args),
        ("11", false, true) => ch11::check(),
        ("11", true, false) => solutions::ch11::run(&args),
        ("11", true, true) => solutions::ch11::check(),
        _ => Err(format!("unknown chapter {chapter}; choose 07..11")),
    }
}
