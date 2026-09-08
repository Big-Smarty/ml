//! Start from Chapter 44's lexical retrieval baseline, then add one Q update.

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

fn q_update(
    old_value: f64,
    reward: f64,
    best_next_value: f64,
    done: bool,
    learning_rate: f64,
    discount_factor: f64,
) -> f64 {
    // TODO: move the old value toward reward plus discounted continuation.
    let _ = (
        old_value,
        reward,
        best_next_value,
        done,
        learning_rate,
        discount_factor,
    );
    todo!("apply the temporal-difference update")
}

fn main() {
    let _guided_todo: fn(f64, f64, f64, bool, f64, f64) -> f64 = q_update;
    let documents = ["rust ownership memory safety", "mars has two moons"];
    let query = "rust memory";
    let best = documents
        .iter()
        .max_by_key(|document| overlap(query, document))
        .expect("fixture contains documents");
    println!("previous checkpoint: retrieved '{best}'");
    println!("Now run cargo test and implement q_update.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_difference_update_matches_worked_example() {
        assert!((q_update(0.2, 1.0, 0.5, false, 0.1, 0.9) - 0.325).abs() < 1e-12);
        assert!((q_update(0.2, 1.0, 99.0, true, 0.1, 0.9) - 0.28).abs() < 1e-12);
    }
}
