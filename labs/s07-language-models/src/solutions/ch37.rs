//! Whitespace is a fixed recipe. Shingles are lowercased sets; repeated shingles count once.
//! Keep the first representative, record removals, then assign entire documents by stable ID.
use crate::{
    corpus::{self, CleanDocument, ParsedDocument, RemovedDuplicate, SplitDocument},
    LabResult,
};
use std::collections::BTreeSet;
fn similarity(left: &str, right: &str) -> f32 {
    let shingles = |text: &str| -> BTreeSet<Vec<String>> {
        let words: Vec<String> = text
            .to_lowercase()
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        words.windows(3).map(|w| w.to_vec()).collect()
    };
    let a = shingles(left);
    let b = shingles(right);
    let union = a.union(&b).count();
    if union == 0 {
        f32::from(left == right)
    } else {
        a.intersection(&b).count() as f32 / union as f32
    }
}
pub fn prepare(
    documents: Vec<ParsedDocument>,
    threshold: f32,
) -> LabResult<(Vec<SplitDocument>, Vec<RemovedDuplicate>)> {
    crate::ensure(
        threshold.is_finite() && (0.0..=1.0).contains(&threshold),
        "threshold must be in 0..=1",
    )?;
    let mut retained: Vec<CleanDocument> = Vec::new();
    let mut removed = Vec::new();
    // ponytail: O(documents²) exact comparison; use MinHash candidate retrieval for large corpora.
    for cleaned in corpus::clean_documents(documents).map_err(|e| e.to_string())? {
        let duplicate = retained.iter().find_map(|prior| {
            let score = similarity(&cleaned.text, &prior.text);
            (cleaned.text == prior.text || score >= threshold).then_some((prior.id, score))
        });
        if let Some((id, score)) = duplicate {
            removed.push((cleaned, id, score));
        } else {
            retained.push(cleaned);
        }
    }
    Ok((
        retained
            .into_iter()
            .map(|document| SplitDocument {
                split: match document.id % 10 {
                    0 => "test",
                    1 => "validation",
                    _ => "train",
                },
                document,
            })
            .collect(),
        removed,
    ))
}
pub fn run(args: &[String]) -> LabResult {
    corpus::run_with(args, prepare)
}
pub fn check() -> LabResult {
    corpus::check_with(prepare)
}
