//! Start from Chapter 44's lexical retrieval baseline, then add one Q update.

fn overlap_score(query: &str, document: &str) -> usize {
    query
        .split_whitespace()
        .filter(|word| document.split_whitespace().any(|token| token == *word))
        .count()
}

fn q_update(old: f64, reward: f64, best_next: f64, alpha: f64, gamma: f64) -> f64 {
    // TODO: move the old value toward reward plus discounted continuation.
    let _ = (old, reward, best_next, alpha, gamma);
    todo!("apply the temporal-difference update")
}

fn main() {
    let _guided_todo: fn(f64, f64, f64, f64, f64) -> f64 = q_update;
    let documents = ["rust ownership memory safety", "mars has two moons"];
    let query = "rust memory";
    let best = documents
        .iter()
        .max_by_key(|document| overlap_score(query, document))
        .expect("fixture contains documents");
    println!("previous checkpoint: retrieved '{best}'");
    println!("Now run cargo test and implement q_update.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_difference_update_matches_worked_example() {
        assert!((q_update(0.2, 1.0, 0.5, 0.1, 0.9) - 0.325).abs() < 1e-12);
    }
}
