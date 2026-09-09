mod ch12;
mod ch13;
mod ch14;
mod ch15;
mod ch16;
mod ch17;
mod ch18;
mod ch19;
mod data;
mod evaluation;
mod solutions {
    pub mod ch12;
    pub mod ch13;
    pub mod ch14;
    pub mod ch15;
    pub mod ch16;
    pub mod ch17;
    pub mod ch18;
    pub mod ch19;
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
        .ok_or("usage: cargo run -- NN [--check] [--solution] [--final for19]")?;
    for arg in &args[1..] {
        if !(["--check", "--solution"].contains(&arg.as_str())
            || arg == "--final" && chapter == "19")
        {
            return Err(format!("unknown argument {arg}"));
        }
    }
    let solution = args.iter().any(|a| a == "--solution");
    let check = args.iter().any(|a| a == "--check");
    match (chapter.as_str(), solution, check) {
        ("12", false, false) => ch12::run(&args[1..]),
        ("12", false, true) => ch12::check(),
        ("12", true, false) => solutions::ch12::run(&args[1..]),
        ("12", true, true) => solutions::ch12::check(),
        ("13", false, false) => ch13::run(&args[1..]),
        ("13", false, true) => ch13::check(),
        ("13", true, false) => solutions::ch13::run(&args[1..]),
        ("13", true, true) => solutions::ch13::check(),
        ("14", false, false) => ch14::run(&args[1..]),
        ("14", false, true) => ch14::check(),
        ("14", true, false) => solutions::ch14::run(&args[1..]),
        ("14", true, true) => solutions::ch14::check(),
        ("15", false, false) => ch15::run(&args[1..]),
        ("15", false, true) => ch15::check(),
        ("15", true, false) => solutions::ch15::run(&args[1..]),
        ("15", true, true) => solutions::ch15::check(),
        ("16", false, false) => ch16::run(&args[1..]),
        ("16", false, true) => ch16::check(),
        ("16", true, false) => solutions::ch16::run(&args[1..]),
        ("16", true, true) => solutions::ch16::check(),
        ("17", false, false) => ch17::run(&args[1..]),
        ("17", false, true) => ch17::check(),
        ("17", true, false) => solutions::ch17::run(&args[1..]),
        ("17", true, true) => solutions::ch17::check(),
        ("18", false, false) => ch18::run(&args[1..]),
        ("18", false, true) => ch18::check(),
        ("18", true, false) => solutions::ch18::run(&args[1..]),
        ("18", true, true) => solutions::ch18::check(),
        ("19", false, false) => ch19::run(&args[1..]),
        ("19", false, true) => ch19::check(),
        ("19", true, false) => solutions::ch19::run(&args[1..]),
        ("19", true, true) => solutions::ch19::check(),
        _ => Err(format!("unknown chapter {chapter}; choose12..19")),
    }
}
