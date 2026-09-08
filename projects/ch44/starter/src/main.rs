fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|x| !x.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}
fn overlap(query: &str, doc: &str) -> usize {
    let d = words(doc);
    words(query).iter().filter(|w| d.contains(w)).count()
}
fn should_abstain(_best: f32, _second: f32, _known_terms: usize) -> bool {
    // TODO: abstain when there is no known evidence or the score margin is too small.
    todo!("implement an evidence gate")
}
fn main() {
    let _guided: fn(f32, f32, usize) -> bool = should_abstain;
    let docs = ["Mars has two moons", "Paris is the capital of France"];
    let q = "capital France";
    let scores: Vec<_> = docs.iter().map(|d| overlap(q, d)).collect();
    println!("checkpoint lexical scores: {scores:?}");
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gate_rejects_unsupported_and_ambiguous() {
        assert!(should_abstain(0.0, 0.0, 0));
        assert!(should_abstain(0.7, 0.69, 2));
        assert!(!should_abstain(0.8, 0.2, 2));
    }
}
