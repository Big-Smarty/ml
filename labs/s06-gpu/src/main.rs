fn main() {
    if let Err(error) = entry() {
        eprintln!("{error}");
        std::process::exit(if error.to_string().starts_with("GOAL_REVIEW_REQUIRED:") {
            3
        } else {
            1
        });
    }
}
fn entry() -> s06_gpu::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let chapter = args
        .first()
        .ok_or("usage: s06-gpu NN [--check] [--solution] [--gpu]")?;
    if !["29", "30", "31", "32"].contains(&chapter.as_str()) {
        return Err("chapter must be 29–32".into());
    }
    if args[1..]
        .iter()
        .any(|a| !["--check", "--solution", "--gpu"].contains(&a.as_str()))
    {
        return Err("unknown option; use --check, --solution, --gpu".into());
    }
    s06_gpu::run(
        chapter,
        args.iter().any(|a| a == "--solution"),
        args.iter().any(|a| a == "--gpu"),
        args.iter().any(|a| a == "--check"),
    )
}
