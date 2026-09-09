//! Supplied, intentionally small experiment CLI: named options always take a value.
use std::collections::BTreeMap;
pub struct Options<'a> {
    values: BTreeMap<&'a str, &'a str>,
}
impl<'a> Options<'a> {
    pub fn parse(args: &'a [String], allowed: &[&str]) -> Result<Self, String> {
        if !args.len().is_multiple_of(2) {
            return Err("each experiment option needs a value".into());
        }
        let mut values = BTreeMap::new();
        for pair in args.chunks_exact(2) {
            let name = pair[0].as_str();
            if !allowed.contains(&name) {
                return Err(format!(
                    "unknown experiment option {name}; allowed: {allowed:?}"
                ));
            }
            if values.insert(name, pair[1].as_str()).is_some() {
                return Err(format!("duplicate option {name}"));
            }
        }
        Ok(Self { values })
    }
    pub fn text(&self, name: &str, default: &'a str) -> &'a str {
        self.values.get(name).copied().unwrap_or(default)
    }
    pub fn number(&self, name: &str, default: f64) -> Result<f64, String> {
        let n = match self.values.get(name) {
            Some(s) => s
                .parse::<f64>()
                .map_err(|_| format!("{name} requires a number"))?,
            None => default,
        };
        if n.is_finite() {
            Ok(n)
        } else {
            Err(format!("{name} must be finite"))
        }
    }
    pub fn count(&self, name: &str, default: usize) -> Result<usize, String> {
        let n = match self.values.get(name) {
            Some(s) => s
                .parse::<usize>()
                .map_err(|_| format!("{name} requires a nonnegative integer"))?,
            None => default,
        };
        if n <= 100_000 {
            Ok(n)
        } else {
            Err(format!(
                "{name} is capped at100000 for these small experiments"
            ))
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_experiments_fail() {
        assert!(Options::parse(&["--rate".into()], &["--rate"]).is_err());
        assert!(Options::parse(&["--unknown".into(), "1".into()], &[]).is_err());
        let args = ["--rate".into(), "NaN".into()];
        assert!(Options::parse(&args, &["--rate"])
            .and_then(|o| o.number("--rate", 0.1))
            .is_err());
    }
}
