mod ch40;
mod ch41;
mod ch42;
mod ch43;
mod ch44;
mod ch45;
mod ch46;
mod solutions;
fn main() {
    if let Err(error) = dispatch() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
fn dispatch() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err("usage: s08-adaptation NN [--check] [--solution] [--bench|--stages]".into());
    }
    let chapter = args.remove(0);
    let check = args.iter().any(|a| a == "--check");
    let solution = args.iter().any(|a| a == "--solution");
    args.retain(|a| a != "--check" && a != "--solution");
    if check && !args.is_empty() {
        return Err("--check accepts no experiment arguments".into());
    }
    match (chapter.as_str(), solution, check) {
        ("40", false, false) => ch40::run(&args),
        ("40", false, true) => ch40::check(),
        ("40", true, false) => solutions::ch40::run(&args),
        ("40", true, true) => solutions::ch40::check(),
        ("41", false, false) => ch41::run(&args),
        ("41", false, true) => ch41::check(),
        ("41", true, false) => solutions::ch41::run(&args),
        ("41", true, true) => solutions::ch41::check(),
        ("42", false, false) => ch42::run(&args),
        ("42", false, true) => ch42::check(),
        ("42", true, false) => solutions::ch42::run(&args),
        ("42", true, true) => solutions::ch42::check(),
        ("43", false, false) => ch43::run(&args),
        ("43", false, true) => ch43::check(),
        ("43", true, false) => solutions::ch43::run(&args),
        ("43", true, true) => solutions::ch43::check(),
        ("44", false, false) => ch44::run(&args),
        ("44", false, true) => ch44::check(),
        ("44", true, false) => solutions::ch44::run(&args),
        ("44", true, true) => solutions::ch44::check(),
        ("45", false, false) => ch45::run(&args),
        ("45", false, true) => ch45::check(),
        ("45", true, false) => solutions::ch45::run(&args),
        ("45", true, true) => solutions::ch45::check(),
        ("46", false, false) => ch46::run(&args),
        ("46", false, true) => ch46::check(),
        ("46", true, false) => solutions::ch46::run(&args),
        ("46", true, true) => solutions::ch46::check(),
        _ => Err("chapter must be 40 through 46".into()),
    }
}
