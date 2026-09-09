// Baselines are runnable earlier/simpler algorithms, not unfinished goal assertions.
#[test]
fn all_ten_offline_baselines_run_and_reject_unknown_arguments() -> Result<(), String> {
    type Run = fn(&[String]) -> Result<(), String>;
    let chapters: [Run; 10] = [
        s09_advanced::ch47::run,
        s09_advanced::ch48::run,
        s09_advanced::ch49::run,
        s09_advanced::ch50::run,
        s09_advanced::ch51::run,
        s09_advanced::ch52::run,
        s09_advanced::ch53::run,
        s09_advanced::ch54::run,
        s09_advanced::ch55::run,
        s09_advanced::ch56::run,
    ];
    for run in chapters {
        run(&[])?;
        assert!(run(&["--not-an-experiment".into()]).is_err());
    }
    Ok(())
}
